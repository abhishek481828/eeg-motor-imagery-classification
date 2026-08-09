"""Conditional GAN Data Augmentation for Synthetic EEG Generation.

Trained strictly on training subjects to avoid data leakage.
Includes synthetic sample validation (shape, amplitude, frequency comparison).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from torch.utils.data import DataLoader

from eeg_mi.utils.logging import get_logger

logger = get_logger("GANAugmentation")


class Generator(nn.Module):
    """Conditional Generator network for EEG signal synthesis."""

    def __init__(self, latent_dim: int, num_channels: int, seq_len: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, num_channels * seq_len),
            nn.Tanh(),
        )
        self.num_channels = num_channels
        self.seq_len = seq_len

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c_emb = self.label_emb(labels)
        x = torch.cat([noise, c_emb], dim=1)
        out = self.fc(x)
        return out.view(out.size(0), self.num_channels, self.seq_len)


class Discriminator(nn.Module):
    """Conditional Discriminator network for synthetic EEG signal validation."""

    def __init__(self, num_channels: int, seq_len: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_channels * seq_len)
        self.fc = nn.Sequential(
            nn.Linear(num_channels * seq_len * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flattened = signal.view(signal.size(0), -1)
        c_emb = self.label_emb(labels)
        x = torch.cat([flattened, c_emb], dim=1)
        return self.fc(x)


class ConditionalGANTrainer:
    """Trainer for Conditional GAN on training subjects only."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        device: torch.device,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Train GAN for one epoch strictly on training subject dataloader."""
        self.generator.train()
        self.discriminator.train()

        loss_g_total = 0.0
        loss_d_total = 0.0

        for real_signals, labels in dataloader:
            batch_size = real_signals.size(0)
            real_signals = real_signals.to(self.device)
            labels = labels.to(self.device)

            real_targets = torch.ones(batch_size, 1, device=self.device)
            fake_targets = torch.zeros(batch_size, 1, device=self.device)

            # Train Discriminator
            self.opt_d.zero_grad()
            noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            fake_signals = self.generator(noise, labels)

            d_loss_real = self.criterion(self.discriminator(real_signals, labels), real_targets)
            d_loss_fake = self.criterion(
                self.discriminator(fake_signals.detach(), labels), fake_targets
            )
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.opt_d.step()

            # Train Generator
            self.opt_g.zero_grad()
            g_loss = self.criterion(self.discriminator(fake_signals, labels), real_targets)
            g_loss.backward()
            self.opt_g.step()

            loss_d_total += d_loss.item() * batch_size
            loss_g_total += g_loss.item() * batch_size

        return loss_g_total / len(dataloader.dataset), loss_d_total / len(dataloader.dataset)


def validate_synthetic_samples(
    real_data: np.ndarray, synthetic_data: np.ndarray, sfreq: float = 160.0
) -> dict[str, any]:
    """Validate synthetic EEG samples against real EEG samples.

    Checks shape consistency, peak-to-peak amplitude bounds,
    and Power Spectral Density (PSD) mean square error.
    """
    assert real_data.shape[1:] == synthetic_data.shape[1:], (
        "Shape mismatch between real and synthetic data"
    )

    real_amp = float(np.mean(np.max(real_data, axis=-1) - np.min(real_data, axis=-1)))
    synth_amp = float(np.mean(np.max(synthetic_data, axis=-1) - np.min(synthetic_data, axis=-1)))

    _, real_psd = welch(real_data, fs=sfreq, axis=-1)
    _, synth_psd = welch(synthetic_data, fs=sfreq, axis=-1)
    psd_mse = float(np.mean((real_psd - synth_psd) ** 2))

    return {
        "real_mean_p2p_amplitude": real_amp,
        "synthetic_mean_p2p_amplitude": synth_amp,
        "psd_mse": psd_mse,
        "shape_matched": True,
    }
