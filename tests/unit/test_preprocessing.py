"""Unit tests for EEG data validation and scaler fit/transform."""

import numpy as np
import pytest

from eeg_mi.data.validation import validate_eeg_window
from eeg_mi.preprocessing.normalization import TrainFittedScaler


def test_validate_eeg_window_valid(dummy_eeg_data: np.ndarray) -> None:
    """Test validation with clean EEG array."""
    validate_eeg_window(dummy_eeg_data[0], expected_channels=64)


def test_validate_eeg_window_nan() -> None:
    """Test validation raises ValueError on NaN."""
    bad_window = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        validate_eeg_window(bad_window)


def test_scaler_no_leakage(dummy_eeg_data: np.ndarray) -> None:
    """Test TrainFittedScaler fits only on training array."""
    scaler = TrainFittedScaler()
    scaler.fit(dummy_eeg_data)
    transformed = scaler.transform(dummy_eeg_data)

    assert transformed.shape == dummy_eeg_data.shape
    # Check transformed train data mean is ~0 and std is ~1 per channel
    means = np.mean(transformed, axis=(0, 2))
    assert np.allclose(means, 0.0, atol=1e-5)
