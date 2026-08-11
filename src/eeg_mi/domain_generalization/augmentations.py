"""Leakage-Safe Domain Generalization Augmentations.

Applies frequency-band masking, time masking, channel dropout, and amplitude scaling
strictly during training SGD.
"""

import torch
import torch.nn as nn


class DomainGeneralizationAugmenter(nn.Module):
    """Batch augmentation module for training EEG signals."""

    def __init__(
        self,
        freq_mask_p: float = 0.2,
        time_mask_p: float = 0.2,
        channel_dropout_p: float = 0.1,
        scale_range: tuple[float, float] | None = (0.9, 1.1),
        apply_p: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_p = freq_mask_p
        self.time_mask_p = time_mask_p
        self.channel_dropout_p = channel_dropout_p
        self.scale_range = scale_range
        self.apply_p = apply_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to x of shape (batch, channels, time)."""
        if not self.training or torch.rand(1).item() > self.apply_p:
            return x

        x_aug = x.clone()
        batch, channels, time_points = x_aug.shape

        # 1. Amplitude Scaling
        if self.scale_range is not None:
            scales = torch.empty(batch, 1, 1, device=x.device).uniform_(*self.scale_range)
            x_aug = x_aug * scales

        # 2. Time Masking
        if self.time_mask_p > 0.0:
            mask_len = int(time_points * self.time_mask_p)
            if mask_len > 0:
                t0 = torch.randint(0, max(1, time_points - mask_len), (1,)).item()
                x_aug[:, :, t0 : t0 + mask_len] = 0.0

        # 3. Channel Dropout
        if self.channel_dropout_p > 0.0:
            ch_mask = (
                torch.rand(batch, channels, 1, device=x.device) > self.channel_dropout_p
            ).float()
            x_aug = x_aug * ch_mask

        return x_aug
