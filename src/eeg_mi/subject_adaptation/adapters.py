"""Subject Adaptation Adapters and Calibrators.

Implements lightweight subject-specific residual adapters, nearest-centroid
prototype calibrators, and temperature scaling for target-subject calibration.
"""

import numpy as np
import torch
import torch.nn as nn


class SubjectAdapter(nn.Module):
    """Lightweight residual bottleneck adapter layer for subject-specific calibration."""

    def __init__(self, feature_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, feature_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual adaptation: x + scale * adapter(x)."""
        return x + self.scale * self.adapter(x)


class PrototypeCalibrator:
    """Nearest-centroid prototype calibrator for target-subject calibration."""

    def __init__(self):
        self.prototypes: dict[int, np.ndarray] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PrototypeCalibrator":
        """Compute mean feature embedding (prototype) for each class in calibration trials."""
        classes = np.unique(labels)
        for c in classes:
            c_mask = labels == c
            self.prototypes[int(c)] = np.mean(features[c_mask], axis=0)
        return self

    def predict_proba(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute Softmax probabilities based on negative Euclidean distance to centroids."""
        assert len(self.prototypes) > 0, "Calibrator must be fitted first!"
        classes = sorted(list(self.prototypes.keys()))
        distances = []

        for c in classes:
            proto = self.prototypes[c]
            # Euclidean distance squared per sample
            dist = np.sum((features - proto) ** 2, axis=1)
            distances.append(-dist / temperature)

        logits = np.column_stack(distances)
        # Softmax over negative distances
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs


class TemperatureScaler(nn.Module):
    """Post-hoc logit temperature scaler for probability calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
