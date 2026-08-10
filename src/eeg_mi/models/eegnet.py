"""Compact EEGNet model architecture for EEG classification.

Adapted from Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
EEG-based Brain-Computer Interfaces" (2018).
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet architecture for multi-channel EEG signals.

    Input tensor shape expected: (batch_size, in_channels, sequence_length)
    """

    def __init__(
        self,
        in_channels: int = 64,
        sequence_length: int = 481,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolution across all EEG channels
        self.depthwise = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(in_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Convolution (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute flatten dimension dynamically using dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, sequence_length)
            feat = self._extract_features(dummy)
            self.flatten_dim = feat.view(1, -1).size(1)

        # Classification Head
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Internal feature extraction pipeline."""
        # Reshape (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects input of shape (batch, channels, time)."""
        feat = self._extract_features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
