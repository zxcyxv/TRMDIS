"""
Loss functions for TRM+DIS coordinate prediction
Combines Huber loss (coordinate accuracy) and Cosine similarity (direction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    AL-GPI loss for DIS-style recursive regression.

    Components:
    1) Contractive Potential Field (cpf)
    2) Continuous Advantage Margin (adv)
    3) Non-degenerate Work (work)
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        adv_margin: float = 0.01,
        adv_weight: float = 1.0,
        work_weight: float = 0.1,
        work_eta: float = 2.0,
        work_eps: float = 0.05,
        work_kappa: float = 10.0,
        x_weight: float = 1.0,
        y_weight: float = 2.38
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.adv_margin = adv_margin
        self.adv_weight = adv_weight
        self.work_weight = work_weight
        self.work_eta = work_eta
        self.work_eps = work_eps
        self.work_kappa = work_kappa
        self.x_weight = x_weight
        self.y_weight = y_weight

    def forward(
        self,
        predictions: torch.Tensor,  # [B, n_sup, 2]
        targets: torch.Tensor        # [B, n_sup, 2]
    ) -> dict:
        """
        Compute combined loss

        Args:
            predictions: [B, n_sup, 2] - Predictions at each supervision step
            targets: [B, n_sup, 2] - Diffusion targets

        Returns:
            Dictionary with loss components:
                - total: Combined loss
                - coord: Huber loss
                - direction: Directional loss
        """
        # 1. Coordinate loss (cpf) with linear step-wise weighting
        n_sup = predictions.size(1)
        step_weights = torch.linspace(
            1.0 / n_sup,
            1.0,
            n_sup,
            device=predictions.device
        )
        y_true = targets[:, -1, :].unsqueeze(1)  # [B, 1, 2]
        if n_sup == 1:
            y0 = y_true.squeeze(1)
        else:
            alpha1 = 1.0 / n_sup
            y0 = (targets[:, 0, :] - alpha1 * y_true.squeeze(1)) / (1.0 - alpha1)

        dist_pred = self._dist_sigma(predictions - y_true)  # [B, n_sup]
        base_dist = self._dist_sigma(y0 - y_true.squeeze(1))  # [B]
        rho = base_dist.unsqueeze(1) * (1.0 - step_weights)  # [B, n_sup]
        slack = torch.clamp(dist_pred - rho, min=0.0)
        coord_loss = self._huber_zero(slack)
        coord_loss = (coord_loss * step_weights).sum() / (step_weights.sum() * coord_loss.size(0))

        # 2) Advantage margin and 3) Work term
        if n_sup == 1:
            y_prev = y0.unsqueeze(1)
        else:
            y_prev = torch.cat([y0.unsqueeze(1), predictions[:, :-1, :]], dim=1)

        delta = predictions - y_prev  # [B, n_sup, 2]
        to_goal = y_true - y_prev     # [B, n_sup, 2]
        to_goal_norm = self._dist_sigma(to_goal).clamp_min(1e-6)
        v_star = to_goal / to_goal_norm.unsqueeze(-1)

        proj = self._dot_sigma(delta, v_star)  # [B, n_sup]
        adv_loss = torch.clamp(self.adv_margin - proj, min=0.0)
        adv_loss = (adv_loss * step_weights).sum() / (step_weights.sum() * adv_loss.size(0))

        move_norm = self._dist_sigma(delta).clamp_min(1e-6)
        work_ratio = move_norm / to_goal_norm

        # Dynamic tolerance gating: compare current error to step-specific target ρ_s
        # If already within target (to_goal_norm < rho), gate → 0 (work term vanishes)
        # If beyond target (to_goal_norm > rho), gate → 1 (enforce movement)
        error_ratio = to_goal_norm / (rho + 1e-6)  # [B, n_sup]
        gate = torch.sigmoid(self.work_kappa * (error_ratio - 1.0))

        work_loss = gate * torch.exp(-self.work_eta * work_ratio)
        early_weights = torch.flip(step_weights, dims=[0])
        work_loss = (work_loss * early_weights).sum() / (early_weights.sum() * work_loss.size(0))

        # 4. Combine losses
        total_loss = coord_loss + self.adv_weight * adv_loss + self.work_weight * work_loss

        return {
            'total': total_loss,
            'coord': coord_loss,
            'direction': adv_loss,
            'work': work_loss
        }

    def _dist_sigma(self, v: torch.Tensor) -> torch.Tensor:
        """Anisotropic Mahalanobis-like distance with field scaling."""
        return torch.sqrt(v[..., 0] ** 2 * self.x_weight + v[..., 1] ** 2 * self.y_weight)

    def _dot_sigma(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Anisotropic inner product."""
        return a[..., 0] * b[..., 0] * self.x_weight + a[..., 1] * b[..., 1] * self.y_weight

    def _huber_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Huber loss to zero with no reduction."""
        return F.huber_loss(x, torch.zeros_like(x), delta=self.huber_delta, reduction='none')


class CTRILoss(nn.Module):
    """
    Contractive Trust Region Improvement Loss (PDF 제안 반영)

    L_CTRI = sum_s omega_s * (
        ||y_s - y*||^2                           # Coordinate loss (Task Accuracy)
        + lambda_mono * R(y_s, y_{s-1}, y*)      # Monotonic improvement constraint
        + lambda_trust * ||y_s - y_{s-1}||^2    # Kinetic/Trust penalty
    )

    R(y_s, y_{s-1}, y*) = ReLU(||y_s - y*|| - γ||y_{s-1} - y*||)
    → γ ≤ 1: 감쇠 계수. 이전 오차의 γ배 이하로 줄어야 함을 강제
    → "확신이 없다면 최소한 이전 위치를 유지(Identity Mapping)"하도록 강제

    Design principles:
    1. Monotonic contraction: Error must decrease by factor γ at each step
    2. Kinetic penalty: Limit step size to prevent overshoot (브레이크 역할)
    3. Step-wise weighting: Later steps have higher weight (ω_s = s²)
    """

    def __init__(
        self,
        lambda_mono: float = 1.0,
        lambda_trust: float = 0.1,
        gamma: float = 0.95,  # 감쇠 계수 (PDF 제안: γ ≤ 1)
        x_weight: float = 2.38,
        y_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_trust = lambda_trust
        self.gamma = gamma  # 단조 개선 감쇠 계수
        self.x_weight = x_weight
        self.y_weight = y_weight

    def _aniso_dist(self, v: torch.Tensor) -> torch.Tensor:
        """Anisotropic distance with field scaling."""
        return torch.sqrt(v[..., 0] ** 2 * self.x_weight + v[..., 1] ** 2 * self.y_weight + 1e-8)

    def forward(
        self,
        predictions: torch.Tensor,  # [B, n_sup, 2]
        targets: torch.Tensor,      # [B, n_sup, 2]
        y_init: torch.Tensor = None # [B, 2] - Initial prediction (optional)
    ) -> dict:
        """
        Compute CTRI loss

        Args:
            predictions: [B, n_sup, 2] - Predictions at each supervision step
            targets: [B, n_sup, 2] - Diffusion targets (y* is targets[:, -1, :])
            y_init: [B, 2] - Initial prediction (if None, derived from targets)

        Returns:
            Dictionary with loss components
        """
        B, n_sup, _ = predictions.shape
        device = predictions.device

        # Ground truth is the final target
        y_true = targets[:, -1, :]  # [B, 2]

        # Derive y_init if not provided (from linear interpolation formula)
        if y_init is None:
            if n_sup == 1:
                y_init = y_true.clone()
            else:
                alpha1 = 1.0 / n_sup
                y_init = (targets[:, 0, :] - alpha1 * y_true) / (1.0 - alpha1 + 1e-8)

        # Step weights: later steps have higher weight (ω_s = s^2 / sum(s^2))
        step_indices = torch.arange(1, n_sup + 1, dtype=torch.float32, device=device)
        step_weights = step_indices ** 2
        step_weights = step_weights / step_weights.sum()

        # Build y_prev: [y_init, y_1, y_2, ..., y_{n_sup-1}]
        y_prev = torch.cat([y_init.unsqueeze(1), predictions[:, :-1, :]], dim=1)  # [B, n_sup, 2]

        # ================================================================
        # 1. Coordinate Loss: ||y_s - y*||^2
        # ================================================================
        coord_errors = self._aniso_dist(predictions - y_true.unsqueeze(1))  # [B, n_sup]
        coord_loss_per_step = coord_errors ** 2  # [B, n_sup]
        coord_loss = (coord_loss_per_step * step_weights).sum(dim=1).mean()

        # ================================================================
        # 2. Monotonic Improvement Constraint (PDF 제안):
        # R = ReLU(||y_s - y*|| - γ||y_{s-1} - y*||)
        # γ ≤ 1: 이전 오차의 γ배 이하로 줄어야 함을 강제
        # ================================================================
        dist_curr = self._aniso_dist(predictions - y_true.unsqueeze(1))  # [B, n_sup]
        dist_prev = self._aniso_dist(y_prev - y_true.unsqueeze(1))       # [B, n_sup]

        # Penalize when current error > γ * previous error
        # γ < 1이면 더 강한 수축 요구 (매 단계 γ배 이하로 줄어야 함)
        mono_violation = F.relu(dist_curr - self.gamma * dist_prev)  # [B, n_sup]
        mono_loss = (mono_violation * step_weights).sum(dim=1).mean()

        # Compute violation rate for monitoring
        mono_violation_rate = (mono_violation > 1e-6).float().mean()

        # ================================================================
        # 3. Kinetic/Trust Penalty (PDF 제안): ||y_s - y_{s-1}||^2
        # 모든 이동에 페널티 부과 (브레이크 역할)
        # 초기 단계의 큰 가중치가 후반부에서 오버슈트를 일으키지 않도록 제어
        # ================================================================
        step_move = self._aniso_dist(predictions - y_prev)  # [B, n_sup]

        # 단순 운동 에너지 페널티: ||y_s - y_{s-1}||^2
        kinetic_loss = step_move ** 2  # [B, n_sup]
        trust_loss = (kinetic_loss * step_weights).sum(dim=1).mean()

        # ================================================================
        # 4. Total Loss
        # ================================================================
        total_loss = coord_loss + self.lambda_mono * mono_loss + self.lambda_trust * trust_loss

        return {
            'total': total_loss,
            'coord': coord_loss,
            'mono': mono_loss,
            'trust': trust_loss,
            'mono_violation_rate': mono_violation_rate,
            # For compatibility with trainer
            'direction': mono_loss,
            'work': trust_loss
        }


class CoordinateLoss(nn.Module):
    """
    Simple coordinate loss (for ablation studies)
    """

    def __init__(self, loss_type: str = 'huber', delta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(delta=delta, reduction='mean')
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Simple coordinate loss

        Args:
            predictions: [B, n_sup, 2]
            targets: [B, n_sup, 2]

        Returns:
            Dictionary with loss
        """
        loss = self.loss_fn(predictions, targets)

        return {
            'total': loss,
            'coord': loss,
            'direction': torch.tensor(0.0, device=predictions.device)
        }
