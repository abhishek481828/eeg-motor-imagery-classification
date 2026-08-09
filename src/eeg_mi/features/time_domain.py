"""Time domain feature extraction (variance, mean, skewness, kurtosis)."""

import numpy as np


def extract_time_domain_features(data: np.ndarray) -> np.ndarray:
    """Extract time domain statistics per channel for data shape (N, C, T)."""
    means = np.mean(data, axis=-1)
    stds = np.std(data, axis=-1)
    variances = np.var(data, axis=-1)
    features = np.concatenate([means, stds, variances], axis=-1)
    return features
