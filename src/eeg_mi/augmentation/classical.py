"""Classical EEG data augmentation (Gaussian noise, time-shifting, scaling)."""

import numpy as np


def add_gaussian_noise(data: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add zero-mean Gaussian noise to EEG signal."""
    noise = np.random.normal(0.0, std, size=data.shape)
    return data + noise


def time_shift(data: np.ndarray, max_shift: int = 10) -> np.ndarray:
    """Apply random time shift along the temporal dimension."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift, axis=-1)
