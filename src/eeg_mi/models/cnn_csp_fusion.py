"""Hybrid CNN + CSP Feature Fusion Model Architecture.

Extracts spatial-temporal features using 1D convolutional layers and fuses
them with log-variance CSP spatial features prior to final classification.
"""

import torch
import torch.nn as nn


class CNN_CSP_Fusion(nn.Module):
    """Hybrid CNN + CSP Feature Fusion Network."""

    def __init__(
        self,
        in_channels: int = 64,
        sequence_length: int = 481,
        num_classes: int = 2,
        csp_components: int = 8,
        filters: list[int] | None = None,
        kernel_size: int = 15,
        dropout: float = 0.25,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]

        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.csp_components = csp_components

        # CNN Feature Extractor
        layers = []
        c_in = in_channels
        for c_out in filters:
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            c_in = c_out
        self.cnn_features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(16)
        self.cnn_dim = filters[-1] * 16

        # Classifier Head (Fuses CNN features + CSP log-var features)
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim + csp_components, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_raw: torch.Tensor, x_csp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_raw: Raw EEG tensor of shape (batch, channels, time)
            x_csp: Pre-calculated CSP features tensor of shape (batch, csp_components)
        """
        feat_cnn = self.cnn_features(x_raw)
        feat_cnn = self.avgpool(feat_cnn)
        feat_cnn = feat_cnn.view(feat_cnn.size(0), -1)

        # Feature Fusion
        fused = torch.cat([feat_cnn, x_csp], dim=1)
        return self.classifier(fused)
