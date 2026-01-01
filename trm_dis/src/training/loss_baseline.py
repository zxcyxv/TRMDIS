"""
Baseline DIS loss (pre AL-GPI).
Coordinate loss: one-way margin to intermediate targets.
Directional loss: cosine alignment between step directions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Baseline combined loss for coordinate regression.

    Components:
    1) One-way margin coordinate loss to diffusion targets
    2) Directional cosine loss (step-to-step)
    """

    def __init__(self, huber_delta: float = 1.0, direction_weight: float = 0.5):
        super().__init__()
        self.huber_delta = huber_delta
        self.direction_weight = direction_weight

    def forward(
        self,
        predictions: torch.Tensor,  # [B, n_sup, 2]
        targets: torch.Tensor        # [B, n_sup, 2]
    ) -> dict:
        n_sup = predictions.size(1)
        step_weights = torch.linspace(
            1.0 / n_sup,
            1.0,
            n_sup,
            device=predictions.device
        )

        y_true = targets[:, -1, :].unsqueeze(1)  # [B, 1, 2]
        dist_pred = torch.norm(predictions - y_true, dim=-1)  # [B, n_sup]
        dist_target = torch.norm(targets - y_true, dim=-1)    # [B, n_sup]
        margin = torch.clamp(dist_pred - dist_target, min=0.0)
        coord_loss = (margin * step_weights).sum() / (step_weights.sum() * margin.size(0))

        direction_loss = torch.tensor(0.0, device=predictions.device)
        if n_sup > 1:
            pred_dirs = predictions[:, 1:, :] - predictions[:, :-1, :]
            target_dirs = targets[:, 1:, :] - targets[:, :-1, :]
            pred_norm = F.normalize(pred_dirs, p=2, dim=-1, eps=1e-6)
            target_norm = F.normalize(target_dirs, p=2, dim=-1, eps=1e-6)
            cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
            direction_loss = (1 - cos_sim).mean()

        total_loss = coord_loss + self.direction_weight * direction_loss

        return {
            'total': total_loss,
            'coord': coord_loss,
            'direction': direction_loss
        }
