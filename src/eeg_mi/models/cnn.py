"""CNN model for spatial-temporal EEG feature extraction."""

import torch
import torch.nn as nn


class EEGCNN(nn.Module):
    """Convolutional Neural Network for EEG classification."""

    def __init__(self, in_channels: int = 64, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.fc = nn.Linear(64 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
