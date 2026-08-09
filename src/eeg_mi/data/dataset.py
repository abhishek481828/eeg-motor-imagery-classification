"""PyTorch Dataset wrapper for preprocessed EEG windows."""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG signal windows and targets."""

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        """Initialize with data shape (N, C, T) and targets (N,)."""
        assert len(data) == len(targets), "Data and targets must have equal length"
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]
