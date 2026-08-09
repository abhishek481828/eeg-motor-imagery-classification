"""Frequency domain Power Spectral Density (PSD) feature extraction."""

import numpy as np
from scipy.signal import welch


def extract_psd_features(data: np.ndarray, sfreq: float = 160.0, bands: dict = None) -> np.ndarray:
    """Extract band power features (e.g. Mu [8-12 Hz] and Beta [13-30 Hz]) per channel."""
    if bands is None:
        bands = {"mu": (8, 12), "beta": (13, 30)}

    n_samples, n_channels, n_times = data.shape
    freqs, psd = welch(data, fs=sfreq, axis=-1)

    feature_list = []
    for _band_name, (f_min, f_max) in bands.items():
        idx_band = np.logical_and(freqs >= f_min, freqs <= f_max)
        band_power = np.mean(psd[:, :, idx_band], axis=-1)  # Shape: (N, C)
        feature_list.append(band_power)

    return np.concatenate(feature_list, axis=-1)  # Shape: (N, C * n_bands)
