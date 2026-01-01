"""
Discrete Diffusion Target Generator for DIS
Generates intermediate targets using linear interpolation
"""
import torch
import torch.nn as nn


class DiffusionTargetGenerator(nn.Module):
    """
    Generates intermediate supervision targets for DIS

    Uses linear interpolation schedule:
        y_s = y_init + (s / n_sup) * (y_true - y_init)

    where s ∈ {1, 2, ..., n_sup}
    """

    def __init__(self, n_sup: int = 6):
        super().__init__()
        self.n_sup = n_sup

        # Precompute interpolation coefficients (alphas)
        # alpha_s = s / n_sup for s in [1, 2, ..., n_sup]
        alphas = torch.linspace(1 / n_sup, 1.0, n_sup)
        self.register_buffer('alphas', alphas)

    def generate_targets(
        self,
        y_init: torch.Tensor,  # [B, 2]
        y_true: torch.Tensor   # [B, 2]
    ) -> torch.Tensor:
        """
        Generate intermediate targets for DIS supervision

        Args:
            y_init: Initial prediction (e.g., field center or zero)
            y_true: Ground truth coordinates (normalized [0, 1])

        Returns:
            targets: [B, n_sup, 2] - Intermediate targets
                y_1, y_2, ..., y_n_sup where y_n_sup = y_true

        Example:
            For n_sup=6:
                y_1 = y_init + 1/6 * (y_true - y_init)
                y_2 = y_init + 2/6 * (y_true - y_init)
                ...
                y_6 = y_init + 6/6 * (y_true - y_init) = y_true
        """
        B = y_init.size(0)
        device = y_init.device

        # Compute delta
        delta = y_true - y_init  # [B, 2]

        # Generate targets for each supervision step
        targets = []
        for alpha in self.alphas:
            y_s = y_init + alpha * delta  # [B, 2]
            targets.append(y_s)

        # Stack: [B, n_sup, 2]
        targets = torch.stack(targets, dim=1)

        return targets

    def verify_targets(self, targets: torch.Tensor, y_true: torch.Tensor) -> bool:
        """
        Verify that targets are monotonically approaching y_true

        Args:
            targets: [B, n_sup, 2]
            y_true: [B, 2]

        Returns:
            True if valid (last target equals y_true and distances decrease)
        """
        # Check last target equals ground truth
        last_target = targets[:, -1, :]  # [B, 2]
        if not torch.allclose(last_target, y_true, atol=1e-6):
            print("WARNING: Last target does not match ground truth!")
            return False

        # Check monotonic decrease in distance
        for s in range(self.n_sup - 1):
            dist_s = torch.norm(targets[:, s, :] - y_true, dim=-1)
            dist_s1 = torch.norm(targets[:, s + 1, :] - y_true, dim=-1)
            if not (dist_s1 <= dist_s + 1e-6).all():
                print(f"WARNING: Distance not decreasing at step {s}")
                return False

        return True
