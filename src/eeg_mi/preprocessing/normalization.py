"""EEG Signal Normalization fitting statistics strictly on training data."""

import numpy as np


class TrainFittedScaler:
    """StandardScaler fitted strictly on training subject data to prevent leakage."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_data: np.ndarray) -> "TrainFittedScaler":
        """Fit scaler on training data shape (N, C, T)."""
        # Axis 0 (samples) and Axis 2 (time) per channel
        self.mean = np.mean(train_data, axis=(0, 2), keepdims=True)
        self.std = np.std(train_data, axis=(0, 2), keepdims=True)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data shape (N, C, T) using fitted mean and std."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted on training data before calling transform")
        return np.asarray((data - self.mean) / self.std)
