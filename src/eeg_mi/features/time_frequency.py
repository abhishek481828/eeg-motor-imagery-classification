"""Time-frequency Wavelet transform features using PyWavelets."""

import numpy as np
import pywt


def extract_wavelet_features(data: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Extract discrete wavelet transform features."""
    n_samples, n_channels, n_times = data.shape
    features = []
    for i in range(n_samples):
        sample_feats = []
        for ch in range(n_channels):
            coeffs = pywt.wavedec(data[i, ch, :], wavelet=wavelet, level=level)
            energy = [np.sum(c**2) for c in coeffs]
            sample_feats.extend(energy)
        features.append(sample_feats)
    return np.array(features)
