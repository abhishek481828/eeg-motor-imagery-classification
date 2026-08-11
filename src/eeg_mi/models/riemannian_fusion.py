"""CNN + Riemannian Tangent Space Feature Fusion Network for EEG Classification.

Concatenates deep 1D-CNN spatio-temporal representations with Riemannian covariance
tangent space feature vectors.
"""

import torch
import torch.nn as nn


class CNNRiemannianFusionModel(nn.Module):
    """Deep CNN + Riemannian Tangent Space Feature Fusion Classifier."""

    def __init__(
        self,
        in_channels: int = 64,
        riemannian_dim: int = 2080,
        num_classes: int = 2,
        cnn_filters: list[int] | None = None,
        kernel_size: int = 15,
        dropout: float = 0.25,
    ):
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [32, 64, 128]

        layers = []
        c_in = in_channels
        for c_out in cnn_filters:
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            c_in = c_out

        self.cnn_features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(16)

        cnn_embedding_dim = cnn_filters[-1] * 16

        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_embedding_dim + riemannian_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_signal: torch.Tensor, x_riemannian: torch.Tensor) -> torch.Tensor:
        """Forward pass taking EEG signal tensor (N, C, T) and Riemannian tangent vector (N, D)."""
        feat_cnn = self.cnn_features(x_signal)
        feat_cnn = self.avgpool(feat_cnn)
        flat_cnn = feat_cnn.view(feat_cnn.size(0), -1)

        fused = torch.cat([flat_cnn, x_riemannian], dim=1)
        out = self.fusion_head(fused)
        return out
