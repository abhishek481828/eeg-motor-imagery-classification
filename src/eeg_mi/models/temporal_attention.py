"""Temporal Attention CNN for EEG Motor Imagery Classification.

Applies lightweight Squeeze-and-Excitation channel and temporal attention
modules to focus on transient motor-imagery ERD/ERS events.
"""

import torch
import torch.nn as nn


class TemporalSqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention over temporal dimension."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)
        weights = self.fc(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * weights


class TemporalAttentionCNN(nn.Module):
    """1D-CNN with integrated Squeeze-and-Excitation Temporal Attention."""

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 2,
        filters: list[int] | None = None,
        kernel_size: int = 15,
        dropout: float = 0.25,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, filters[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True),
            TemporalSqueezeExcitation(filters[0]),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(filters[0], filters[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters[1]),
            nn.ReLU(inplace=True),
            TemporalSqueezeExcitation(filters[1]),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(filters[1], filters[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters[2]),
            nn.ReLU(inplace=True),
            TemporalSqueezeExcitation(filters[2]),
            nn.AdaptiveAvgPool1d(16),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(filters[2] * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        flattened = x.view(x.size(0), -1)
        out = self.fc(flattened)
        return out
