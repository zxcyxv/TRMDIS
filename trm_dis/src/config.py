"""
Configuration file for TRM+DIS Football Coordinate Prediction Model
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model configuration for TRM+DIS architecture"""

    # Data parameters
    max_seq_len: int = 16           # Maximum sequence length (last N actions)
    field_x_max: float = 105.0      # Field width in meters
    field_y_max: float = 68.0       # Field height in meters

    # Embedding parameters
    d_model: int = 256              # Hidden dimension
    n_action_types: int = 26        # Number of action types
    n_result_types: int = 9         # Number of result types (8 + 1 for NaN)
    dropout: float = 0.1            # Dropout rate

    # Transformer parameters
    n_layers: int = 2               # Number of Transformer layers (TRM paper)
    n_heads: int = 8                # Number of attention heads
    d_ff: int = 1024                # Feedforward dimension (SwiGLU)

    # DIS Recursion parameters (CRITICAL!)
    n_sup: int = 4                  # Number of supervision steps
    T: int = 1                      # External cycles (NO no-grad in DIS!)
    n: int = 4                      # Internal latent updates per supervision step

    # Diffusion parameters
    diffusion_type: Literal["linear", "gaussian", "hybrid"] = "hybrid"
    hybrid_noise_scale: float = 0.05    # Noise scale for hybrid diffusion (training only)
    gaussian_sigma_init: float = 0.3    # Initial sigma for gaussian diffusion
    gaussian_sigma_final: float = 0.0   # Final sigma for gaussian diffusion

    # Spectral Normalization (for contraction mapping)
    use_spectral_norm: bool = True      # Apply spectral norm to attention/FFN layers

    # Dynamic Stopping Gate (prevent oscillation near target)
    use_dynamic_stopping: bool = True   # Enable dynamic stopping gate
    stopping_threshold: float = 0.02    # Normalized distance threshold
    stopping_kappa: float = 20.0        # Gate sigmoid steepness

    # Gaussian Denoising Training (PDF 제안)
    use_denoising_training: bool = True     # 학습 시 입력에 노이즈 주입
    denoising_sigma_init: float = 0.1       # 초기 노이즈 스케일 (초기 단계)
    denoising_sigma_final: float = 0.01     # 최종 노이즈 스케일 (후반 단계)

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    ema_decay: float = 0.999
    max_epochs: int = 100
    warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Loss function selection
    use_ctri_loss: bool = True      # Use CTRI loss (False = use AL-GPI)

    # CTRI Loss parameters (PDF 제안 반영)
    ctri_lambda_mono: float = 1.0           # Monotonic improvement constraint weight
    ctri_lambda_trust: float = 0.1          # Kinetic/Trust penalty weight
    ctri_gamma: float = 0.95                # 감쇠 계수 γ ≤ 1 (PDF 제안)

    # AL-GPI Loss weights (used when use_ctri_loss=False)
    huber_delta: float = 1.0        # Huber loss delta for cpf slack
    adv_margin: float = 0.01        # Minimum projected progress margin
    adv_weight: float = 1.0         # Weight for advantage margin term
    work_weight: float = 0.1        # Weight for work term
    work_eta: float = 2.0           # Exponential sharpness for work term
    work_eps: float = 0.012         # Distance threshold for work gating
    work_kappa: float = 10.0        # Gating steepness for work term

    # Anisotropic distance weights (shared by both loss functions)
    x_weight: float = 2.38          # Anisotropic distance weight for x (105/68)^2
    y_weight: float = 1.0           # Anisotropic distance weight for y

    # Data paths
    train_csv_path: str = "/workspace/TRMDIS/open_track1/train.csv"
    test_csv_path: str = "/workspace/TRMDIS/open_track1/test.csv"
    test_dir_path: str = "/workspace/TRMDIS/open_track1/test"
    sample_submission_path: str = "/workspace/TRMDIS/open_track1/sample_submission.csv"
    cond_feature_path: str = "/workspace/TRMDIS/Test3/open_track1/film_smoe_cond_features.npz"
    use_cond_features: bool = True
    cond_dim: int = 196  # cls_out(128) + fourier_seq(64) + router_logits(4)
    use_cls_only: bool = False
    cls_dim: int = 128

    # Output paths
    checkpoint_dir: str = "/workspace/TRMDIS/trm_dis/outputs/checkpoints"
    log_dir: str = "/workspace/TRMDIS/trm_dis/outputs/logs"
    submission_path: str = "/workspace/TRMDIS/trm_dis/outputs/submission.csv"

    # Device
    device: str = "cuda"            # cuda or cpu
    num_workers: int = 4            # DataLoader workers

    # Validation
    val_split: float = 0.1          # Validation split ratio
    val_episodes: int = 1000        # Number of validation episodes

    # Logging
    log_interval: int = 100         # Log every N batches
    save_interval: int = 1          # Save checkpoint every N epochs

    def __post_init__(self):
        """Validate configuration"""
        assert self.T == 1, "DIS uses T=1 (no no-grad cycles!)"
        assert self.n > 0, "n must be positive"
        assert self.n_sup > 0, "n_sup must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class ExperimentConfig:
    """Experiment-specific configuration"""

    name: str = "trm_dis_baseline"
    seed: int = 42
    debug_mode: bool = False        # Use small subset for debugging
    debug_samples: int = 100        # Number of samples in debug mode

    # Experiment variants
    use_ema: bool = True
    use_direction_loss: bool = True
    use_diffusion_targets: bool = True

    # Model config
    model: ModelConfig = field(default_factory=ModelConfig)
