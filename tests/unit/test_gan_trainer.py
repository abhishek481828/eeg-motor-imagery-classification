"""Unit tests for GAN Augmentation & Validation (Phase 8)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_mi.augmentation.gan import (
    ConditionalGANTrainer,
    Discriminator,
    Generator,
    validate_synthetic_samples,
)
from eeg_mi.data.dataset import EEGDataset


def test_gan_trainer_epoch(dummy_eeg_data: np.ndarray, dummy_targets: np.ndarray) -> None:
    """Test running 1 epoch of Conditional GAN training on dummy training dataset."""
    dataset = EEGDataset(dummy_eeg_data, dummy_targets)
    loader = DataLoader(dataset, batch_size=4)

    gen = Generator(latent_dim=10, num_channels=64, seq_len=480, num_classes=4)
    disc = Discriminator(num_channels=64, seq_len=480, num_classes=4)
    device = torch.device("cpu")

    trainer = ConditionalGANTrainer(gen, disc, device, lr_g=0.001, lr_d=0.001)
    loss_g, loss_d = trainer.train_epoch(loader)

    assert loss_g > 0.0
    assert loss_d > 0.0


def test_validate_synthetic_samples(dummy_eeg_data: np.ndarray) -> None:
    """Test synthetic signal shape, amplitude, and PSD validation checks."""
    synthetic_data = dummy_eeg_data + np.random.normal(0, 0.1, dummy_eeg_data.shape)
    metrics = validate_synthetic_samples(dummy_eeg_data, synthetic_data)

    assert metrics["shape_matched"] is True
    assert "real_mean_p2p_amplitude" in metrics
    assert "synthetic_mean_p2p_amplitude" in metrics
    assert "psd_mse" in metrics
