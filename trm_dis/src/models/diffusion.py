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


class HybridDiffusionTargetGenerator(nn.Module):
    """
    Hybrid Diffusion Target Generator with Gaussian Denoising

    Combines linear interpolation with Gaussian noise:
        y_s = y_init + alpha_s * (y_true - y_init) + sigma_s * epsilon

    where:
        - alpha_s = s / n_sup (linear interpolation)
        - sigma_s = noise_scale * (1 - s / n_sup) (decreasing noise)
        - epsilon ~ N(0, I)

    Purpose:
        - Training: Add noise to simulate inference-time prediction errors
        - This teaches the model to denoise/recover from off-manifold predictions
        - Reduces exposure bias between training and inference
    """

    def __init__(
        self,
        n_sup: int = 4,
        noise_scale: float = 0.05,
        noise_decay: str = 'linear'
    ):
        super().__init__()
        self.n_sup = n_sup
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay

        # Precompute interpolation coefficients
        alphas = torch.linspace(1 / n_sup, 1.0, n_sup)
        self.register_buffer('alphas', alphas)

        # Precompute noise weights (decreasing over steps)
        if noise_decay == 'linear':
            noise_weights = torch.linspace(1.0, 0.0, n_sup)
        elif noise_decay == 'cosine':
            t = torch.linspace(0, 1, n_sup)
            noise_weights = 0.5 * (1 + torch.cos(t * 3.14159))
        else:
            noise_weights = torch.ones(n_sup)
        self.register_buffer('noise_weights', noise_weights)

    def generate_targets(
        self,
        y_init: torch.Tensor,  # [B, 2]
        y_true: torch.Tensor   # [B, 2]
    ) -> torch.Tensor:
        """
        Generate intermediate targets with optional noise

        Args:
            y_init: Initial prediction [B, 2]
            y_true: Ground truth coordinates [B, 2]

        Returns:
            targets: [B, n_sup, 2]
        """
        B = y_init.size(0)
        device = y_init.device
        delta = y_true - y_init

        targets = []
        for s, alpha in enumerate(self.alphas):
            # Linear interpolation
            y_s = y_init + alpha * delta

            # Add noise during training only
            if self.training and self.noise_scale > 0:
                noise = torch.randn(B, 2, device=device)
                noise_weight = self.noise_weights[s]
                y_s = y_s + self.noise_scale * noise_weight * noise

                # Clip to valid range [0, 1]
                y_s = torch.clamp(y_s, 0.0, 1.0)

            targets.append(y_s)

        return torch.stack(targets, dim=1)


class GaussianDiffusionTargetGenerator(nn.Module):
    """
    Pure Gaussian Diffusion Target Generator

    Generates targets by adding decreasing noise to ground truth:
        y_s = y_true + sigma_s * epsilon

    where sigma_s decreases from sigma_init to sigma_final

    This is a denoising approach where the model learns to
    progressively remove noise to reach the target.
    """

    def __init__(
        self,
        n_sup: int = 4,
        sigma_init: float = 0.3,
        sigma_final: float = 0.0,
        schedule: str = 'cosine'
    ):
        super().__init__()
        self.n_sup = n_sup
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.schedule = schedule

        # Precompute sigma schedule
        sigmas = self._compute_sigma_schedule()
        self.register_buffer('sigmas', sigmas)

    def _compute_sigma_schedule(self) -> torch.Tensor:
        """Compute noise schedule (decreasing)"""
        t = torch.linspace(0, 1, self.n_sup)

        if self.schedule == 'linear':
            sigmas = self.sigma_init + t * (self.sigma_final - self.sigma_init)
        elif self.schedule == 'cosine':
            # Cosine: fast initial decrease, slow final decrease
            sigmas = self.sigma_final + 0.5 * (self.sigma_init - self.sigma_final) * (1 + torch.cos(t * 3.14159))
        elif self.schedule == 'quadratic':
            sigmas = self.sigma_final + (self.sigma_init - self.sigma_final) * (1 - t) ** 2
        else:
            sigmas = torch.ones(self.n_sup) * self.sigma_init

        return sigmas

    def generate_targets(
        self,
        y_init: torch.Tensor,  # [B, 2] - not used, kept for interface compatibility
        y_true: torch.Tensor   # [B, 2]
    ) -> torch.Tensor:
        """
        Generate noisy targets centered on ground truth

        Args:
            y_init: Initial prediction (ignored in this implementation)
            y_true: Ground truth coordinates [B, 2]

        Returns:
            targets: [B, n_sup, 2]
        """
        B = y_true.size(0)
        device = y_true.device

        targets = []
        for sigma in self.sigmas:
            if self.training and sigma > 0:
                noise = torch.randn(B, 2, device=device)
                y_s = y_true + sigma * noise
                y_s = torch.clamp(y_s, 0.0, 1.0)
            else:
                y_s = y_true.clone()

            targets.append(y_s)

        return torch.stack(targets, dim=1)
