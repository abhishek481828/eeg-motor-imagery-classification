"""Subject-Invariant Channel Normalization Transformers.

Fits channel statistics (mean, std, median, IQR) strictly on training data
and applies linear scaling to transform evaluation sets without test leakage.
"""

import numpy as np


class PerChannelZScoreScaler:
    """Channel-wise Z-score scaler fitted on training dataset."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PerChannelZScoreScaler":
        """Fit channel-wise mean and std on X of shape (num_epochs, channels, time)."""
        # Collapse epochs and time points per channel
        # X shape: (N, C, T) -> axis (0, 2)
        self.means = np.mean(X, axis=(0, 2), keepdims=True)
        self.stds = np.std(X, axis=(0, 2), keepdims=True)
        self.stds = np.maximum(self.stds, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize X using fitted training statistics."""
        assert self.means is not None and self.stds is not None, "Scaler must be fitted first!"
        return (X - self.means) / self.stds


class SubjectRobustScaler:
    """Channel-wise Robust Scaler using Median and Interquartile Range (IQR)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.medians: np.ndarray | None = None
        self.iqrs: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SubjectRobustScaler":
        """Fit channel-wise median and IQR on X of shape (num_epochs, channels, time)."""
        # Collapse epochs and time points per channel
        # Flatten across N and T per channel
        N, C, T = X.shape
        X_flat = np.transpose(X, (1, 0, 2)).reshape(C, N * T)

        q25 = np.percentile(X_flat, 25, axis=1, keepdims=True)
        q75 = np.percentile(X_flat, 75, axis=1, keepdims=True)
        med = np.median(X_flat, axis=1, keepdims=True)

        self.medians = med.reshape(1, C, 1)
        iqr = q75 - q25
        self.iqrs = np.maximum(iqr, self.eps).reshape(1, C, 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using fitted training median and IQR."""
        assert self.medians is not None and self.iqrs is not None, "Scaler must be fitted first!"
        return (X - self.medians) / self.iqrs
