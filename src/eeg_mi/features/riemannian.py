"""Riemannian Geometry Covariance Matrix Feature Extractor."""

import numpy as np


def compute_covariance_matrices(data: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrices for signal data shape (N, C, T)."""
    n_samples, n_channels, n_times = data.shape
    covs = np.zeros((n_samples, n_channels, n_channels))
    for i in range(n_samples):
        covs[i] = np.cov(data[i])
    return covs
