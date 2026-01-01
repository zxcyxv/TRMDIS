"""
TRM+DIS Main Model
Implements Tiny Recursive Models with Deep Improvement Supervision

CRITICAL DESIGN POINTS (from paper re-analysis):
1. T=1, n=2 loop (NO no-grad cycles!)
2. Sequence dimension preserved (h, y, z all [B, seq_len, d_model])
3. Direct assignment (z = f_θ(...), NOT z = z + delta)
4. Input combination: ADDITION (x + y + z), NOT concatenation!
5. z update: f_θ(x + y + z) - WITH input x
6. y update: f_θ(y + z) - WITHOUT input x (naturally 2 inputs)

ARCHITECTURAL REFINEMENTS:
7. y_init from last action's start coordinates (physical context)
8. Time-step embedding for supervision step awareness (DIS paper Fig.3)
9. Sigmoid bounding for coordinate outputs [0, 1]

Reference: TRM paper Section 2.2, HRM paper Eq. 1
  zL ← fL(zL + zH + x)   # addition, not concatenation!
  zH ← fH(zL + zH)       # no x in y update
"""
import torch
import torch.nn as nn
from .embeddings import EpisodeEmbedding
from .transformer import TransformerBackbone, RMSNorm
from .diffusion import DiffusionTargetGenerator


class TRMDIS(nn.Module):
    """
    TRM+DIS model for football coordinate prediction

    Architecture (corrected to match paper):
    1. Embed episode → Transformer → h [B, seq_len, d_model]
    2. Initialize y from last action's start_xy (physical prior)
    3. For s in [1..n_sup]:
        - Inject time-step embedding for step s
        - For i in [1..n]: z = f_θ(x + y + z + step_emb)  (latent reasoning)
        - y = f_θ(y + z + step_emb)  (answer refinement)
        - coords = sigmoid(output_head(y))[:, -1, :]  (bounded output)

    Key insight from TRM paper Section 4.3:
    "since z ← fL(x + y + z) contains x but y ← fH(y + z) does not contain x,
    the task to achieve is directly specified by the inclusion or lack of x"

    Architectural Refinements:
    - y_init: Start from last action's (start_x, start_y) projected to d_model
    - Step embedding: Integer-based timestep (DIS paper recommends over continuous)
    - Sigmoid output: Bounds predictions to valid field coordinates [0, 1]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_action_types: int = 26,
        n_result_types: int = 9,
        n_sup: int = 6,
        n: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 16,
        cls_dim: int = 128,
        use_cls_only: bool = False,
        use_cond_features: bool = False,
        cond_dim: int = 196
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sup = n_sup
        self.n = n
        self.use_cls_only = use_cls_only
        self.use_cond_features = use_cond_features

        # Embedding layer
        self.embedding = EpisodeEmbedding(
            d_model=d_model,
            n_action_types=n_action_types,
            n_result_types=n_result_types,
            dropout=dropout
        )

        # Optional CLS-only input projection
        self.cls_proj = nn.Linear(cls_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # Diffusion target generator
        self.diffusion = DiffusionTargetGenerator(n_sup=n_sup)

        # ============================================================
        # REFINEMENT 1: y_init projection from start coordinates
        # Projects (start_x, start_y) → d_model for physical initialization
        # ============================================================
        self.y_init_proj = nn.Sequential(
            nn.Linear(2, d_model),
            RMSNorm(d_model)
        )

        # ============================================================
        # REFINEMENT 2: Time-step embedding (DIS paper recommends integer-based)
        # Each supervision step s ∈ {0, 1, ..., n_sup-1} gets unique embedding
        # Helps f_θ distinguish "direction finding" (early) vs "fine-tuning" (late)
        # ============================================================
        self.step_embedding = nn.Embedding(n_sup, d_model)

        # Shared network f_θ for both z and y updates
        # Input: d_model (after addition: x+y+z+step or y+z+step)
        # Output: d_model - DIRECT assignment
        #
        # TRM paper baseline uses a 2-layer Transformer for recursive reasoning.
        self.f_theta = TransformerBackbone(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # Output head: y [B, seq_len, d_model] → coordinates [B, seq_len, 2]
        # ============================================================
        # REFINEMENT 3: Sigmoid applied after this layer for [0,1] bounding
        # ============================================================
        self.output_head = nn.Linear(d_model, 2)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(
        self,
        continuous: torch.Tensor,  # [B, seq_len, 4]
        categorical: torch.Tensor,  # [B, seq_len, 3]
        mask: torch.Tensor,         # [B, seq_len]
        targets: torch.Tensor = None,  # [B, 2] - ground truth (training only)
        cls_out: torch.Tensor = None,  # [B, cls_dim] - optional CLS features
        cond_seq: torch.Tensor = None  # [B, seq_len, cond_dim] - optional conditioning features
    ) -> dict:
        """
        Forward pass

        Args:
            continuous: [B, seq_len, 4] - (start_x, start_y, time_norm, padding_flag)
            categorical: [B, seq_len, 3] - (type_id, result_id, is_home)
            mask: [B, seq_len]
            targets: [B, 2] - ground truth coordinates (training only)

        Returns:
            Dictionary:
                - predictions: [B, n_sup, 2] - predictions at each supervision step
                - final_pred: [B, 2] - final prediction
                - diff_targets: [B, n_sup, 2] - diffusion targets (training only)
        """
        B, seq_len, _ = continuous.shape
        device = continuous.device

        # 1. Encode episode sequence (no separate backbone)
        h = self.embedding(continuous, categorical)  # [B, seq_len, d_model]
        if self.use_cls_only and cls_out is not None:
            h = self.cls_proj(cls_out).unsqueeze(1).expand(B, seq_len, self.d_model)
        if self.use_cond_features and cond_seq is not None:
            cond = self.cond_proj(cond_seq)
            h = h + cond

        # KEEP SEQUENCE DIMENSION! (NO mean pooling)
        # h: [B, seq_len, d_model]

        # ============================================================
        # REFINEMENT 1: Initialize y from last action's start coordinates
        # Physical prior: end position is strongly correlated with start position
        # ============================================================
        # Extract last action's (start_x, start_y) from continuous features
        last_start_xy = continuous[:, -1, :2]  # [B, 2] - (start_x, start_y) normalized

        # Project to d_model and broadcast to sequence
        y_init_emb = self.y_init_proj(last_start_xy)  # [B, d_model]
        y = y_init_emb.unsqueeze(1).expand(B, seq_len, self.d_model)  # [B, seq_len, d_model]

        # z: latent reasoning state (initialized to zero)
        z = torch.zeros(B, seq_len, self.d_model, device=device)

        # 3. Get initial coordinate prediction for diffusion targets
        # coords_init uses the physical prior (last action's start position)
        coords_init = last_start_xy  # [B, 2] - already normalized

        # 4. Generate diffusion targets (training only)
        diff_targets = None
        if self.training and targets is not None:
            diff_targets = self.diffusion.generate_targets(coords_init, targets)

        predictions = []

        # 5. Deep Improvement Supervision Loop
        # DIS: T=1, n=2 (NO no-grad cycles!)
        #
        # Paper reference (TRM Section 2.2, HRM Eq. 1):
        #   zL ← fL(zL + zH + x)   # ADDITION
        #   zH ← fH(zL + zH)       # no x in y update
        for s in range(self.n_sup):
            # ============================================================
            # REFINEMENT 2: Time-step embedding injection
            # Integer-based step embedding (DIS paper recommends over continuous)
            # Helps f_θ distinguish early (direction) vs late (fine-tuning) steps
            # ============================================================
            step_idx = torch.tensor([s], device=device)
            step_emb = self.step_embedding(step_idx)  # [1, d_model]
            step_emb = step_emb.unsqueeze(0).expand(B, seq_len, self.d_model)  # [B, seq_len, d_model]

            # n=2 latent reasoning steps (WITH input x=h)
            # z ← f_θ(h + y + z + step_emb) - ADDITION with step awareness
            for i in range(self.n):
                z = self.f_theta(h + y + z + step_emb, mask)  # [B, seq_len, d_model]

            # Answer refinement (WITHOUT input x, WITH step_emb)
            # y ← f_θ(y + z + step_emb) - naturally excludes input
            y = self.f_theta(y + z + step_emb, mask)  # [B, seq_len, d_model]

            # ============================================================
            # REFINEMENT 3: Sigmoid bounding for [0, 1] coordinate output
            # Prevents predictions outside valid field boundaries
            # ============================================================
            coords = torch.sigmoid(self.output_head(y))  # [B, seq_len, 2] in [0, 1]
            coords_last = coords[:, -1, :]  # [B, 2]
            predictions.append(coords_last)

            # Detach for next supervision step
            if s < self.n_sup - 1:
                y = y.detach()
                z = z.detach()

        # Stack predictions: [B, n_sup, 2]
        predictions = torch.stack(predictions, dim=1)

        return {
            'predictions': predictions,
            'final_pred': predictions[:, -1, :],  # Last supervision step
            'diff_targets': diff_targets
        }

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def verify_gradient_flow(self) -> dict:
        """
        Verify that all parameters receive gradients

        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'params_without_grad': []
        }

        for name, param in self.named_parameters():
            stats['total_params'] += 1
            if param.grad is None:
                stats['params_without_grad'].append(name)
            else:
                stats['params_with_grad'] += 1

        return stats
