"""
Exponential Moving Average (EMA) for model parameters
Provides stable predictions by averaging parameter updates
"""
import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average wrapper for PyTorch models

    Usage:
        model = MyModel()
        ema = EMA(model, decay=0.999)

        # Training loop
        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update()  # Update EMA parameters

        # Inference
        with ema.average_parameters():
            predictions = model(test_data)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = smoother, typical: 0.999)
        """
        self.model = model
        self.decay = decay

        # Create shadow parameters (deep copy of current parameters)
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update EMA parameters after each training step

        Formula: shadow = decay * shadow + (1 - decay) * param
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def apply_shadow(self):
        """Apply EMA parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters (after inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @torch.no_grad()
    def average_parameters(self):
        """
        Context manager for using EMA parameters

        Example:
            with ema.average_parameters():
                output = model(input)
        """
        return _EMAContext(self)

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class _EMAContext:
    """Context manager for temporarily applying EMA parameters"""

    def __init__(self, ema: EMA):
        self.ema = ema

    def __enter__(self):
        self.ema.apply_shadow()

    def __exit__(self, *args):
        self.ema.restore()
