"""Integration test for model training loop execution."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.models.factory import create_model
from eeg_mi.training.trainer import Trainer


def test_training_loop_integration(
    dummy_eeg_data: np.ndarray, dummy_targets: np.ndarray, tmp_path: Path
) -> None:
    """Test running a 2-epoch training loop end-to-end using Trainer engine."""
    # Create 2-class dummy dataset for unit test of Trainer engine
    dataset = EEGDataset(dummy_eeg_data, np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=4)

    model = create_model("cnn_lstm", num_channels=64, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    ckpt_path = tmp_path / "test_ckpt.pt"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=ckpt_path,
        patience=5,
    )

    history = trainer.fit(train_loader=loader, val_loader=loader, epochs=2)

    assert len(history["train_loss"]) == 2
    assert ckpt_path.exists()
