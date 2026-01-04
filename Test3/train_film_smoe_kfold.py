"""
FiLM + Spatial MoE Hybrid Transformer - K-Fold OOF Version

5-Fold Out-of-Fold training to eliminate data leakage in CLS feature extraction.

SHAP 분석 결과를 반영한 영역 조건부 아키텍처:
1. FiLM (Feature-wise Linear Modulation): Magnitude Explosion 해결 (dt_7: 60x)
2. Spatial MoE: Sign Reversal 해결 (res_id_7: -0.29 → +0.68)
3. Auxiliary Task: Boundary distance 예측으로 공간 이해 강화

Architecture:
- SpatialConditioningNetwork: (start_x_7, start_y_7) → γ, β
- FiLMLayer: h_out = γ ⊙ norm(h) + β
- SpatialRouter: G(pos) ∈ [0,1] for expert weighting
- ExpertHead (x2): In-field / Boundary specialists
- Final: y = μ + σ ⊙ r (Heteroscedastic output)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'open_track1'
FIELD_X, FIELD_Y = 105, 68
K = 8


# =============================================================================
# Base Components (from heteroscedastic model)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = self.gate(normed) * attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FourierFeatureLayer(nn.Module):
    """Random Fourier Features: γ(v) = [cos(2πBv), sin(2πBv)]"""
    def __init__(self, input_dim, mapping_size=32, scale=5.0):
        super().__init__()
        torch.manual_seed(42)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        self.output_dim = mapping_size * 2
        coord_max = torch.tensor([105.0, 68.0, 105.0, 68.0])
        self.register_buffer('coord_max', coord_max)

    def forward(self, x):
        x_norm = x / self.coord_max
        x_proj = 2 * np.pi * torch.matmul(x_norm, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# =============================================================================
# FiLM Components
# =============================================================================

class SpatialConditioningNetwork(nn.Module):
    """
    FiLM Generator: (start_x_7, start_y_7) → γ, β

    위치 정보를 기반으로 Transformer의 특성 추출을 제어하는
    scale(γ)과 shift(β) 파라미터를 실시간 생성
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model

        # Coordinate normalization
        self.register_buffer('coord_max', torch.tensor([105.0, 68.0]))

        # Fourier features for position
        self.fourier_dim = 32
        torch.manual_seed(43)  # Different seed from main Fourier layer
        B = torch.randn(2, self.fourier_dim) * 5.0
        self.register_buffer('B', B)

        # Input: 2 raw + 64 Fourier = 66
        input_dim = 2 + self.fourier_dim * 2

        # Conditioning MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Separate heads for gamma and beta
        self.gamma_head = nn.Linear(hidden_dim, d_model)
        self.beta_head = nn.Linear(hidden_dim, d_model)

        # Initialize: gamma=1 (identity), beta=0 (no shift)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos: [batch, 2] - (start_x_7, start_y_7)
        Returns:
            gamma: [batch, d_model] - scale parameters
            beta: [batch, d_model] - shift parameters
        """
        # Normalize
        pos_norm = pos / self.coord_max  # [batch, 2]

        # Fourier encoding
        proj = 2 * np.pi * torch.matmul(pos_norm, self.B)  # [batch, fourier_dim]
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [batch, 64]

        # Concatenate
        x = torch.cat([pos_norm, fourier], dim=-1)  # [batch, 66]

        # MLP
        h = self.mlp(x)  # [batch, hidden_dim]

        # Generate FiLM parameters
        gamma = self.gamma_head(h)  # [batch, d_model]
        beta = self.beta_head(h)    # [batch, d_model]

        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: h_out = γ ⊙ norm(h) + β

    Magnitude Explosion 해결: dt_7의 60배 증폭을 γ로 제어
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model]
            gamma: [batch, d_model]
            beta: [batch, d_model]
        Returns:
            h_out: [batch, seq_len, d_model]
        """
        h_norm = self.norm(h)

        # Broadcast to sequence dimension
        gamma = gamma.unsqueeze(1)  # [batch, 1, d_model]
        beta = beta.unsqueeze(1)    # [batch, 1, d_model]

        return gamma * h_norm + beta


# =============================================================================
# Spatial MoE Components
# =============================================================================

class SpatialRouter(nn.Module):
    """
    Spatial Routing: G(pos) → [0,1]

    G(pos) = 1: Boundary Expert
    G(pos) = 0: In-field Expert

    Sign Reversal 해결: 영역별로 다른 전문가 가중치 적용

    Temperature Annealing: 초기에 soft (높은 temp) → 점진적으로 hard (낮은 temp)
    """
    def __init__(self, hidden_dim: int = 64, fourier_dim: int = 32, fourier_scale: float = 10.0):
        super().__init__()

        self.register_buffer('coord_max', torch.tensor([105.0, 68.0]))

        # Fourier features for high-frequency boundary sensitivity.
        self.fourier_dim = fourier_dim
        B = torch.randn(6, fourier_dim) * fourier_scale
        self.register_buffer('B', B)

        # Router MLP (input: 6 raw + 2*fourier_dim)
        input_dim = 6 + 2 * fourier_dim
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Fixed temperature (will be controlled externally via annealing)
        self.base_temperature = 1.0

    def compute_boundary_distances(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute normalized distances to all 4 boundaries."""
        x, y = pos[:, 0], pos[:, 1]

        dist_left = x / 105.0
        dist_right = (105.0 - x) / 105.0
        dist_bottom = y / 68.0
        dist_top = (68.0 - y) / 68.0

        return torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1)

    def forward(self, pos: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            pos: [batch, 2] - (start_x_7, start_y_7)
            temperature: temperature for sigmoid (lower = sharper decisions)
        Returns:
            gate: [batch, 1] - boundary expert weight
        """
        pos_norm = pos / self.coord_max
        boundary_dists = self.compute_boundary_distances(pos)

        x = torch.cat([pos_norm, boundary_dists], dim=-1)  # [batch, 6]
        proj = 2 * torch.pi * (x @ self.B)
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        x = torch.cat([x, fourier], dim=-1)

        logit = self.router(x) / (temperature + 1e-6)
        gate = torch.sigmoid(logit)

        return gate


class ExpertHead(nn.Module):
    """
    Specialized Expert Head: (μ, log_var, residual)

    Expert A: In-field 전문가
    Expert B: Boundary 전문가
    """
    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Variance head
        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Residual head (larger capacity)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: mu, log_var, residual [batch, 2] each"""
        shared = self.shared(h)
        return self.mean_head(shared), self.var_head(shared), self.residual_head(shared)


class AuxiliaryBoundaryHead(nn.Module):
    """
    Auxiliary Task: Boundary Distance + Zone Classification

    모델이 공간 구조를 더 잘 이해하도록 유도
    """
    def __init__(self, d_model: int = 128):
        super().__init__()

        # Distance regression
        self.dist_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Zone classification (4 classes)
        self.zone_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: dist_pred [batch, 1], zone_logits [batch, 4]"""
        return self.dist_head(h), self.zone_head(h)


# =============================================================================
# Main Model: FiLM + Spatial MoE Transformer
# =============================================================================

class FiLMSpatialMoETransformer(nn.Module):
    """
    FiLM + Spatial MoE Hybrid Transformer

    1. FourierFeatureLayer: 좌표 인코딩
    2. Transformer + FiLM: 위치 조건부 특성 추출
    3. Spatial MoE: 영역별 전문가 혼합
    4. Heteroscedastic Output: y = μ + σ ⊙ r
    """
    def __init__(
        self,
        input_size: int = 24,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        fourier_mapping_size: int = 32,
        fourier_scale: float = 5.0,
        film_hidden_dim: int = 64,
        router_hidden_dim: int = 32,
    ):
        super().__init__()

        self.d_model = d_model
        self.K = K

        # Coordinate indices
        self.coord_indices = [0, 1, 14, 15]  # start_x, start_y, end_x, end_y
        self.start_x_idx = 0
        self.start_y_idx = 1

        # Fourier Feature Layer
        self.fourier_layer = FourierFeatureLayer(
            len(self.coord_indices), fourier_mapping_size, fourier_scale
        )

        # Total input size
        total_input_size = input_size + self.fourier_layer.output_dim  # 24 + 64 = 88

        # Input projection
        self.input_proj = nn.Linear(total_input_size, d_model)

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # FiLM components
        self.conditioning_net = SpatialConditioningNetwork(d_model, film_hidden_dim)
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model) for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Spatial MoE components
        self.spatial_router = SpatialRouter(router_hidden_dim)
        self.expert_infield = ExpertHead(d_model, dropout)
        self.expert_boundary = ExpertHead(d_model, dropout)

        # Auxiliary head
        self.aux_head = AuxiliaryBoundaryHead(d_model)

    def extract_position(self, x: torch.Tensor) -> torch.Tensor:
        """Extract (start_x_7, start_y_7) from input sequence."""
        return x[:, -1, [self.start_x_idx, self.start_y_idx]]

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, 8, 24] - input sequence
            temperature: router temperature (lower = sharper routing)
            return_all: whether to return all outputs
        """
        batch_size, seq_len, n_features = x.shape

        # 1. Extract spatial position
        pos = self.extract_position(x)  # [batch, 2]

        # 2. Generate FiLM parameters
        gamma, beta = self.conditioning_net(pos)

        # 3. Fourier encoding
        coord_features = x[:, :, self.coord_indices]
        fourier_features = self.fourier_layer(coord_features)
        x = torch.cat([fourier_features, x], dim=-1)

        # 4. Input projection
        x = self.input_proj(x)

        # 5. Add CLS token and position embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 6. Transformer layers with FiLM
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.film_layers[i](x, gamma, beta)

        # 7. Final norm and CLS extraction
        x = self.final_norm(x)
        cls_out = x[:, 0, :]

        # 8. Spatial routing with temperature
        gate = self.spatial_router(pos, temperature=temperature)  # [batch, 1]

        # 9. Expert predictions
        mu_A, log_var_A, res_A = self.expert_infield(cls_out)
        mu_B, log_var_B, res_B = self.expert_boundary(cls_out)

        # 10. Mixture of experts (no dropout - router learns directly)
        mu = gate * mu_B + (1 - gate) * mu_A
        log_var = gate * log_var_B + (1 - gate) * log_var_A
        residual = gate * res_B + (1 - gate) * res_A

        # 11. Heteroscedastic output
        sigma = torch.exp(0.5 * log_var)
        y_final = mu + sigma * residual

        # 12. Auxiliary predictions
        aux_dist, aux_zone = self.aux_head(cls_out)

        outputs = {
            'y_final': y_final,
            'mu': mu,
            'log_var': log_var,
            'sigma': sigma,
            'residual': residual,
            'gate': gate,
            'gamma': gamma,
            'beta': beta,
            'cls_out': cls_out,
            'aux_dist': aux_dist,
            'aux_zone': aux_zone,
            'mu_A': mu_A,
            'mu_B': mu_B,
        }

        if return_all:
            return outputs
        return outputs


# =============================================================================
# Loss Function
# =============================================================================

def gaussian_nll_loss(y_true, mu, log_var):
    """Gaussian NLL: L = exp(-s) * ||y - μ||² + s"""
    precision = torch.exp(-log_var)
    mse = (y_true - mu) ** 2
    loss = precision * mse + log_var
    return loss.mean()


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    boundary_dist: torch.Tensor,
    boundary_zone: torch.Tensor,
    w_nll: float = 1.0,
    w_final: float = 5.0,
    w_aux_dist: float = 0.5,
    w_aux_zone: float = 0.5,
    w_gate: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss with Gate Supervision.

    Gate Supervision: BCE(G(pos), I(zone ≠ 0))
    - In-field (zone=0) → target=0
    - Boundary (zone=1,2,3) → target=1
    """

    # 1. NLL Loss
    nll_loss = gaussian_nll_loss(y_true, outputs['mu'], outputs['log_var'])

    # 2. MSE Final
    mse_final = F.mse_loss(outputs['y_final'], y_true)

    # 3. Auxiliary Distance (Huber)
    aux_dist_loss = F.smooth_l1_loss(outputs['aux_dist'], boundary_dist)

    # 4. Auxiliary Zone (CE)
    aux_zone_loss = F.cross_entropy(outputs['aux_zone'], boundary_zone, label_smoothing=0.1)

    # 5. Gate Supervision Loss (KEY FIX for Router Collapse)
    # Target: 0 for In-field (zone=0), 1 for Boundary (zone=1,2,3)
    gate = outputs['gate']
    gate_target = (boundary_zone != 0).float().unsqueeze(1)  # [batch, 1]
    gate_loss = F.binary_cross_entropy(gate, gate_target)

    # Total
    total_loss = (
        w_nll * nll_loss +
        w_final * mse_final +
        w_aux_dist * aux_dist_loss +
        w_aux_zone * aux_zone_loss +
        w_gate * gate_loss
    )

    loss_dict = {
        'nll': nll_loss.item(),
        'mse_final': mse_final.item(),
        'aux_dist': aux_dist_loss.item(),
        'aux_zone': aux_zone_loss.item(),
        'gate': gate_loss.item(),
        'total': total_loss.item(),
    }

    return total_loss, loss_dict


# =============================================================================
# Training Scheduler
# =============================================================================

class TrainingScheduler:
    """
    Simplified Training Schedule with Temperature Annealing

    Phase 1 (Warm-up, 10 epochs): High temperature (soft routing), gate supervision
    Phase 2 (Main, 10-100 epochs): Temperature annealing 2.0 → 0.5
    Phase 3 (Fine-tune, 100+ epochs): Low temperature (sharp routing)

    Gate Supervision is always on to prevent router collapse.
    """
    def __init__(self, warmup_epochs=10, anneal_epochs=90):
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.anneal_end = warmup_epochs + anneal_epochs

    def get_temperature(self, epoch: int) -> float:
        """
        Temperature annealing: 2.0 → 0.5

        High temp = soft decisions (early training)
        Low temp = sharp decisions (late training)
        """
        if epoch < self.warmup_epochs:
            return 2.0  # Soft during warm-up
        elif epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return 2.0 - 1.5 * progress  # 2.0 → 0.5
        else:
            return 0.5  # Sharp routing

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Loss weights with gate supervision always on.
        """
        if epoch < self.warmup_epochs:
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.3, 'w_aux_zone': 0.3,
                'w_gate': 2.0  # Strong gate supervision early
            }
        elif epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.3 + 0.2 * progress,  # 0.3 → 0.5
                'w_aux_zone': 0.3 + 0.2 * progress,  # 0.3 → 0.5
                'w_gate': 2.0 - 1.0 * progress  # 2.0 → 1.0
            }
        else:
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.5, 'w_aux_zone': 0.5,
                'w_gate': 1.0
            }


# =============================================================================
# Data Preparation
# =============================================================================

def create_boundary_labels(end_x, end_y):
    """Zone labels: 0=In-field, 1=Top-out, 2=Bottom-out, 3=Goal-line"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def compute_boundary_distance(end_x, end_y):
    """Minimum distance to nearest boundary (normalized)."""
    dist_left = end_x
    dist_right = FIELD_X - end_x
    dist_bottom = end_y
    dist_top = FIELD_Y - end_y
    min_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top))
    return min_dist / (FIELD_X / 2)  # Normalize


def prepare_sequence_data(df, k=K):
    base_features = ['start_x', 'start_y', 'dt', 'ep_idx_norm', 'x_zone', 'lane',
                     'dist_to_goal', 'angle_to_goal', 'type_id', 'res_id', 'is_home',
                     'pressure_x_weight', 'is_zone14', 'angle_visible']
    masked_features = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed',
                       'action_angle', 'action_progress', 'action_dist', 'action_lateral']

    n_samples = len(df)
    n_base = len(base_features)
    n_features = n_base + len(masked_features)
    X_seq = np.zeros((n_samples, k, n_features))

    feature_indices = {
        'start_x': [0], 'start_y': [1], 'x_zone': [4], 'lane': [5],
        'dist_to_goal': [6], 'angle_to_goal': [7],
        'end_x': [n_base], 'end_y': [n_base + 1],
        'dx': [n_base + 2], 'dy': [n_base + 3]
    }

    for t in range(k):
        for j, feat in enumerate(base_features):
            col = f'{feat}_{t}'
            if col in df.columns:
                X_seq[:, t, j] = df[col].fillna(0).values
        for j, feat in enumerate(masked_features):
            col = f'{feat}_{t}'
            if col in df.columns and t < k - 1:
                X_seq[:, t, n_base + j] = df[col].fillna(0).values

    return X_seq, feature_indices


def augment_data(X, y_coords, boundary_dist, boundary_zone, feature_indices):
    """Y-flip augmentation with auxiliary targets."""
    X_all = [X.copy()]
    y_coords_all = [y_coords.copy()]
    boundary_dist_all = [boundary_dist.copy()]
    boundary_zone_all = [boundary_zone.copy()]

    X_yflip = X.copy()
    for idx in feature_indices.get('start_y', []): X_yflip[:, :, idx] = FIELD_Y - X_yflip[:, :, idx]
    for idx in feature_indices.get('end_y', []): X_yflip[:, :, idx] = FIELD_Y - X_yflip[:, :, idx]
    for idx in feature_indices.get('dy', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]
    for idx in feature_indices.get('lane', []):
        lane = X_yflip[:, :, idx].copy()
        X_yflip[:, :, idx] = np.where(lane == 0, 2, np.where(lane == 2, 0, lane))
    for idx in feature_indices.get('angle_to_goal', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    # Zone labels: Top-out ↔ Bottom-out swap
    zone_yflip = boundary_zone.copy()
    zone_yflip = np.where(zone_yflip == 1, 2, np.where(zone_yflip == 2, 1, zone_yflip))

    X_all.append(X_yflip)
    y_coords_all.append(y_coords_yflip)
    boundary_dist_all.append(boundary_dist.copy())  # Distance is symmetric
    boundary_zone_all.append(zone_yflip)

    return (np.concatenate(X_all), np.concatenate(y_coords_all),
            np.concatenate(boundary_dist_all), np.concatenate(boundary_zone_all))


# =============================================================================
# K-Fold OOF Helper Functions
# =============================================================================

def create_episode_kfold_splits(df, n_splits=5, random_state=42):
    """
    Create K-fold splits at episode level (not row level).
    Keeps all K=8 actions from each episode together to prevent leakage.

    Args:
        df: DataFrame with 'game_episode' column
        n_splits: Number of folds (default 5)
        random_state: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get unique episodes
    unique_episodes = df['game_episode'].unique()

    # Create mapping: episode → row indices
    episode_to_indices = df.groupby('game_episode').indices

    # Split episodes (not rows)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_splits = []

    for fold_idx, (train_ep_idx, val_ep_idx) in enumerate(kf.split(unique_episodes)):
        train_episodes = unique_episodes[train_ep_idx]
        val_episodes = unique_episodes[val_ep_idx]

        # Map episodes back to row indices
        train_indices = np.concatenate([episode_to_indices[ep] for ep in train_episodes])
        val_indices = np.concatenate([episode_to_indices[ep] for ep in val_episodes])

        fold_splits.append((train_indices, val_indices))

    return fold_splits


def validate_fold_splits(fold_splits, df):
    """
    Validate no episode overlap between train/val folds.
    Raises assertion error if data leakage detected.
    """
    for i, (train_idx, val_idx) in enumerate(fold_splits):
        train_eps = set(df.iloc[train_idx]['game_episode'])
        val_eps = set(df.iloc[val_idx]['game_episode'])
        overlap = train_eps & val_eps
        assert len(overlap) == 0, f"Fold {i}: {len(overlap)} episode overlap detected!"
    log(f"✓ Validated {len(fold_splits)} folds - no episode overlap")


# =============================================================================
# Main Training
# =============================================================================

def main():
    log("=" * 70)
    log("FiLM + Spatial MoE - K-Fold OOF Training")
    log("  - 5-Fold Cross-Validation")
    log("  - Episode-level splits (no leakage)")
    log("  - Out-of-Fold feature extraction")
    log("=" * 70)

    # Load full dataset
    log("\n[1] Loading data...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    X_seq, feature_indices = prepare_sequence_data(df_all, K)
    y_coords = df_all[['target_end_x', 'target_end_y']].values

    # Auxiliary targets
    boundary_zone = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)
    boundary_dist = compute_boundary_distance(df_all['target_end_x'].values, df_all['target_end_y'].values)

    log(f"  Total episodes: {df_all['game_episode'].nunique()}")
    log(f"  Total samples: {len(df_all)}")

    # Create K-fold splits
    log("\n[2] Creating episode-level K-fold splits...")
    fold_splits = create_episode_kfold_splits(df_all, n_splits=5, random_state=42)
    validate_fold_splits(fold_splits, df_all)

    # Initialize OOF storage
    n_samples = len(df_all)
    oof_cls_features = np.zeros((n_samples, 128), dtype=np.float32)
    oof_fourier_seq = np.zeros((n_samples, 8, 64), dtype=np.float32)
    oof_router_logits = np.zeros((n_samples, 4), dtype=np.float32)
    oof_predictions = np.zeros((n_samples, 2), dtype=np.float32)
    fold_metadata = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device: {device}")

    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        log(f"\n{'='*70}")
        log(f"FOLD {fold_idx}/4")
        log(f"{'='*70}")

        # Set fold-specific seed
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)

        # Extract fold data (BEFORE augmentation)
        X_train_orig = X_seq[train_idx]
        X_val = X_seq[val_idx]
        y_coords_train_orig = y_coords[train_idx]
        y_coords_val = y_coords[val_idx]
        boundary_dist_train_orig = boundary_dist[train_idx]
        boundary_dist_val = boundary_dist[val_idx]
        boundary_zone_train_orig = boundary_zone[train_idx]
        boundary_zone_val = boundary_zone[val_idx]

        log(f"  Train: {len(X_train_orig)} samples ({len(set(df_all.iloc[train_idx]['game_episode']))} episodes)")
        log(f"  Val: {len(X_val)} samples ({len(set(df_all.iloc[val_idx]['game_episode']))} episodes)")

        # Augment TRAINING only
        log(f"  Augmenting training data (Y-flip)...")
        X_train, y_coords_train, boundary_dist_train, boundary_zone_train = augment_data(
            X_train_orig, y_coords_train_orig, boundary_dist_train_orig, boundary_zone_train_orig, feature_indices
        )
        log(f"  Train (augmented): {len(X_train)} samples")

        # Fit scaler on TRAINING only
        n_samples_train, n_steps, n_features = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples_train, n_steps, n_features)
        X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), n_steps, n_features)

        # Normalize coordinates
        coord_mean = np.array([FIELD_X / 2, FIELD_Y / 2])
        coord_std = np.array([FIELD_X / 2, FIELD_Y / 2])
        y_coords_train_norm = (y_coords_train - coord_mean) / coord_std
        y_coords_val_norm = (y_coords_val - coord_mean) / coord_std

        # Create tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
        boundary_dist_train_t = torch.FloatTensor(boundary_dist_train.reshape(-1, 1)).to(device)
        boundary_zone_train_t = torch.LongTensor(boundary_zone_train).to(device)

        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t, boundary_dist_train_t, boundary_zone_train_t)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        # Initialize FRESH model (critical for no leakage!)
        log(f"  Initializing fresh model for fold {fold_idx}...")
        model = FiLMSpatialMoETransformer(
            input_size=n_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.2,
            fourier_mapping_size=32,
            fourier_scale=5.0,
            film_hidden_dim=64,
            router_hidden_dim=32,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Parameters: {n_params:,}")

        # Training components
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        training_scheduler = TrainingScheduler(warmup_epochs=10, anneal_epochs=90)

        best_dist = float('inf')
        best_state = None
        patience, patience_counter = 50, 0

        # Training loop
        log(f"  Training fold {fold_idx}...")
        for epoch in range(200):
            model.train()
            temperature = training_scheduler.get_temperature(epoch)
            loss_weights = training_scheduler.get_loss_weights(epoch)

            for X_batch, y_batch, dist_batch, zone_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch, temperature=temperature, return_all=True)
                loss, loss_dict = compute_loss(outputs, y_batch, dist_batch, zone_batch, **loss_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_t, temperature=0.05, return_all=True)
                final_np = outputs['y_final'].cpu().numpy() * coord_std + coord_mean
                final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()

            lr_scheduler.step(final_dist)

            if final_dist < best_dist:
                best_dist = final_dist
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                phase = "Warm-up" if epoch < 10 else ("Anneal" if epoch < 100 else "Fine-tune")
                log(f"    Epoch {epoch+1:3d} [{phase:8s}] T={temperature:.2f}: Final={final_dist:.2f}m")

            if patience_counter >= patience:
                log(f"    Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_state)
        log(f"  Fold {fold_idx} Best Distance: {best_dist:.4f}m")

        # Save fold checkpoint
        checkpoint_path = DATA_DIR / f'film_smoe_fold{fold_idx}.pt'
        torch.save({
            'fold_idx': fold_idx,
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'coord_mean': coord_mean,
            'coord_std': coord_std,
            'best_dist': best_dist,
            'train_episodes': len(set(df_all.iloc[train_idx]['game_episode'])),
            'val_episodes': len(set(df_all.iloc[val_idx]['game_episode'])),
        }, checkpoint_path)
        log(f"  Saved: {checkpoint_path}")

        # Extract OOF features for validation fold
        log(f"  Extracting OOF features for fold {fold_idx}...")
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t, temperature=0.05, return_all=True)

            # Store OOF features
            oof_cls_features[val_idx] = outputs['cls_out'].cpu().numpy()

            # Fourier features
            coord_features = X_val_t[:, :, [0, 1, 2, 3]]  # Approximate coord indices
            fourier_features = model.fourier_layer(coord_features)
            oof_fourier_seq[val_idx] = fourier_features.cpu().numpy()

            # Router logits
            oof_router_logits[val_idx] = outputs['aux_zone'].cpu().numpy()

            # Predictions (for validation)
            final_np = outputs['y_final'].cpu().numpy() * coord_std + coord_mean
            oof_predictions[val_idx] = final_np

        # Store metadata
        fold_metadata.append({
            'fold_idx': fold_idx,
            'best_dist': float(best_dist),
            'train_episodes': len(set(df_all.iloc[train_idx]['game_episode'])),
            'val_episodes': len(set(df_all.iloc[val_idx]['game_episode'])),
        })

        # Clean up GPU
        del model, optimizer
        torch.cuda.empty_cache()

    # Overall OOF performance
    overall_oof_dist = np.sqrt(((oof_predictions - y_coords) ** 2).sum(axis=1)).mean()
    fold_distances = [m['best_dist'] for m in fold_metadata]

    log(f"\n{'='*70}")
    log("OVERALL OOF PERFORMANCE")
    log(f"{'='*70}")
    log(f"  Mean fold distance: {np.mean(fold_distances):.4f}m")
    log(f"  Std fold distance:  {np.std(fold_distances):.4f}m")
    log(f"  Overall OOF distance: {overall_oof_dist:.4f}m")
    for i, m in enumerate(fold_metadata):
        log(f"  Fold {i}: {m['best_dist']:.4f}m")

    # Save OOF features
    oof_npz_path = DATA_DIR / 'film_smoe_cond_features_oof.npz'
    np.savez(
        oof_npz_path,
        episode_ids=df_all['game_episode'].values,
        cls_out=oof_cls_features,
        fourier_seq=oof_fourier_seq,
        router_logits=oof_router_logits,
    )
    log(f"\n  Saved: {oof_npz_path}")

    # Save metadata
    metadata = {
        'n_folds': 5,
        'random_state': 42,
        'fold_distances': [float(d) for d in fold_distances],
        'overall_oof_distance': float(overall_oof_dist),
        'total_episodes': int(df_all['game_episode'].nunique()),
        'total_samples': int(len(df_all)),
        'folds': fold_metadata,
    }
    with open(DATA_DIR / 'film_smoe_kfold_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    log(f"  Saved: {DATA_DIR / 'film_smoe_kfold_metadata.json'}")

    log(f"\n{'='*70}")
    log("K-Fold OOF Training Complete!")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
