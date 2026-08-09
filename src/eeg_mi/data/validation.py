"""Dataset validation checks for array shapes, NaNs, and infinite values."""

import numpy as np


def validate_eeg_window(window: np.ndarray, expected_channels: int | None = None) -> None:
    """Validate single EEG segment array for NaNs, infinity, or shape issues."""
    if np.isnan(window).any():
        raise ValueError("EEG window contains NaN values")
    if np.isinf(window).any():
        raise ValueError("EEG window contains infinite values")
    if window.size == 0:
        raise ValueError("EEG window is empty")
    if expected_channels is not None and window.shape[0] != expected_channels:
        raise ValueError(f"Channel mismatch: expected {expected_channels}, got {window.shape[0]}")
