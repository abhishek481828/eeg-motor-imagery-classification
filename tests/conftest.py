"""Pytest fixtures for unit and integration testing."""

import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_eeg_data() -> np.ndarray:
    """Return dummy EEG numpy array shape (10, 64, 480)."""
    np.random.seed(42)
    return np.random.randn(10, 64, 480)


@pytest.fixture
def dummy_targets() -> np.ndarray:
    """Return dummy targets array shape (10,)."""
    return np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])


@pytest.fixture
def dummy_eeg_tensor() -> torch.Tensor:
    """Return PyTorch tensor shape (4, 64, 480)."""
    torch.manual_seed(42)
    return torch.randn(4, 64, 480)
