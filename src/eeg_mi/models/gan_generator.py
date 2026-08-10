"""WGAN-GP Synthetic EEG Trial Generator Module (Train-Only Augmentation).

Generates synthetic EEG motor imagery epochs of shape (64 channels, 481 time points).
Trained strictly on training subjects S001-S077.
"""

import torch
import torch.nn as nn


class EEGGenerator(nn.Module):
    """Deep Convolutional Generator for EEG signals."""

    def __init__(self, latent_dim: int = 64, num_channels: int = 64, time_points: int = 481):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.time_points = time_points

        self.init_size = time_points // 4
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=time_points),
            nn.Conv1d(64, num_channels, kernel_size=5, padding=2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        img = self.conv_blocks(out)
        return img


class EEGDiscriminator(nn.Module):
    """Deep Convolutional Discriminator for EEG signals."""

    def __init__(self, num_channels: int = 64, time_points: int = 481):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_wgan_gp(
    generator: EEGGenerator,
    discriminator: EEGDiscriminator,
    X_train: torch.Tensor,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.0002,
) -> EEGGenerator:
    """Train WGAN-GP strictly on X_train tensor."""
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = torch.utils.data.TensorDataset(X_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator.to(device)
    discriminator.to(device)

    for epoch in range(1, epochs + 1):
        for batch in loader:
            real_x = batch[0].to(device)
            b_sz = real_x.size(0)

            # Train Discriminator
            d_opt.zero_grad()
            z = torch.randn(b_sz, generator.latent_dim, device=device)
            fake_x = generator(z).detach()

            d_loss = -torch.mean(discriminator(real_x)) + torch.mean(discriminator(fake_x))
            d_loss.backward()
            d_opt.step()

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(b_sz, generator.latent_dim, device=device)
            gen_x = generator(z)
            g_loss = -torch.mean(discriminator(gen_x))
            g_loss.backward()
            g_opt.step()

    generator.eval()
    return generator
