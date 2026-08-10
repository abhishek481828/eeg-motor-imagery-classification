"""Safe Real-Signal EEG Data Augmentations (Train-Only).

Applied strictly to training batches during SGD to prevent overfitting
without altering physiological timing structure or test integrity.
"""

import torch
import torch.nn as nn


class EEGAugmenter(nn.Module):
    """Batch-level real-signal EEG data augmenter."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        max_shift_samples: int = 15,
        channel_dropout_p: float = 0.05,
        apply_p: float = 0.5,
    ):
        super().__init__()
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.max_shift_samples = max_shift_samples
        self.channel_dropout_p = channel_dropout_p
        self.apply_p = apply_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to batch tensor of shape (batch, channels, time)."""
        if not self.training or torch.rand(1).item() > self.apply_p:
            return x

        x_aug = x.clone()
        batch, num_channels, num_time = x_aug.shape

        # 1. Amplitude Scaling
        if self.scale_range is not None:
            scales = torch.empty(batch, 1, 1, device=x.device).uniform_(*self.scale_range)
            x_aug = x_aug * scales

        # 2. Gaussian Noise Injection
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

        # 3. Temporal Circular Shift
        if self.max_shift_samples > 0:
            shift = torch.randint(-self.max_shift_samples, self.max_shift_samples + 1, (1,)).item()
            x_aug = torch.roll(x_aug, shifts=shift, dims=2)

        # 4. Random Channel Dropout
        if self.channel_dropout_p > 0:
            mask = torch.rand(batch, num_channels, 1, device=x.device) > self.channel_dropout_p
            x_aug = x_aug * mask.float()

        return x_aug
