"""Integration test for Dataset & DataLoader creation pipeline."""

import numpy as np
from torch.utils.data import DataLoader

from eeg_mi.data.dataset import EEGDataset


def test_dataset_dataloader_integration(
    dummy_eeg_data: np.ndarray, dummy_targets: np.ndarray
) -> None:
    """Test creating dataset and loading batches via PyTorch DataLoader."""
    dataset = EEGDataset(dummy_eeg_data, dummy_targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batches = list(loader)
    assert len(batches) == 3  # 10 samples / batch_size 4 -> 3 batches
    x_b, y_b = batches[0]
    assert x_b.shape == (4, 64, 480)
    assert y_b.shape == (4,)
