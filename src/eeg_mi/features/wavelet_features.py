"""Wavelet Time-Frequency Feature Extraction Module.

Extracts Discrete Wavelet Transform (DWT) multi-resolution features from EEG signals.
Features per channel & sub-band:
  - Sub-band energy / power
  - Mean & Standard deviation
  - Relative energy proportion
  - Spectral entropy
"""

import numpy as np
import pywt


class WaveletFeatureExtractor:
    """Wavelet time-frequency feature extractor for multi-channel EEG tensors."""

    def __init__(self, wavelet: str = "db4", level: int = 4):
        self.wavelet = wavelet
        self.level = level

    def transform_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """Extract wavelet features for a single epoch of shape (channels, time)."""
        num_channels, _ = epoch.shape
        feats = []

        for c in range(num_channels):
            sig = epoch[c]
            coeffs = pywt.wavedec(sig, self.wavelet, level=self.level)

            total_energy = sum(np.sum(np.square(c_arr)) for c_arr in coeffs) + 1e-12

            for c_arr in coeffs:
                energy = np.sum(np.square(c_arr))
                rel_energy = energy / total_energy
                mean_val = np.mean(c_arr)
                std_val = np.std(c_arr)

                # Sub-band entropy
                p = np.abs(c_arr) / (np.sum(np.abs(c_arr)) + 1e-12)
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p + 1e-12))

                feats.extend([energy, rel_energy, mean_val, std_val, entropy])

        return np.array(feats, dtype=np.float32)

    def transform_dataset(self, X: np.ndarray) -> np.ndarray:
        """Extract wavelet feature matrix for dataset of shape (epochs, channels, time)."""
        num_epochs = X.shape[0]
        feature_list = [self.transform_epoch(X[i]) for i in range(num_epochs)]
        return np.vstack(feature_list)
