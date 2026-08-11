"""Multi-Scale Temporal Convolutional Neural Network for EEG Motor Imagery.

Extracts features across multiple parallel receptive fields (k=7, 15, 31, 63)
to capture distinct oscillatory EEG dynamics across frequency sub-bands (theta, mu, beta, gamma).
"""

import torch
import torch.nn as nn


class MultiScaleConvBranch(nn.Module):
    """Single temporal convolution branch with specific kernel size."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.25
    ):
        super().__init__()
        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class MultiScaleCNN(nn.Module):
    """Multi-Scale 1D CNN with parallel receptive fields and feature fusion."""

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 2,
        branch_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.25,
    ):
        super().__init__()
        if branch_channels is None:
            branch_channels = [16, 32, 64]
        if kernel_sizes is None:
            kernel_sizes = [7, 15, 31, 63]

        self.branches = nn.ModuleList(
            [
                MultiScaleConvBranch(in_channels, branch_channels[0], k_sz, dropout=dropout)
                for k_sz in kernel_sizes
            ]
        )

        num_branches = len(kernel_sizes)
        merged_channels = branch_channels[0] * num_branches

        self.fused_layers = nn.Sequential(
            nn.Conv1d(merged_channels, branch_channels[1], kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(branch_channels[1], branch_channels[2], kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(branch_channels[2] * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass shape: (batch, channels, time) -> (batch, num_classes)."""
        branch_outs = [branch(x) for branch in self.branches]
        concat_features = torch.cat(branch_outs, dim=1)
        fused = self.fused_layers(concat_features)
        flattened = fused.view(fused.size(0), -1)
        out = self.classifier(flattened)
        return out
