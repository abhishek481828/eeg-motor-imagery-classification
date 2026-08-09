"""Unit tests for PyTorch models (CNN, LSTM, CNN-LSTM hybrid, and GAN)."""

import torch

from eeg_mi.augmentation.gan import Discriminator, Generator
from eeg_mi.models.factory import create_model


def test_cnn_lstm_forward(dummy_eeg_tensor: torch.Tensor) -> None:
    """Test CNN-LSTM model forward pass and output shape."""
    model = create_model("cnn_lstm", num_channels=64, num_classes=4)
    out = model(dummy_eeg_tensor)
    assert out.shape == (4, 4)


def test_cnn_forward(dummy_eeg_tensor: torch.Tensor) -> None:
    """Test CNN model forward pass."""
    model = create_model("cnn", num_channels=64, num_classes=4)
    out = model(dummy_eeg_tensor)
    assert out.shape == (4, 4)


def test_gan_generator_and_discriminator() -> None:
    """Test GAN Generator and Discriminator tensor shapes."""
    batch_size = 4
    latent_dim = 100
    num_channels = 64
    seq_len = 480
    num_classes = 4

    gen = Generator(latent_dim, num_channels, seq_len, num_classes)
    disc = Discriminator(num_channels, seq_len, num_classes)

    noise = torch.randn(batch_size, latent_dim)
    labels = torch.tensor([0, 1, 2, 3])

    fake_signal = gen(noise, labels)
    assert fake_signal.shape == (batch_size, num_channels, seq_len)

    pred = disc(fake_signal, labels)
    assert pred.shape == (batch_size, 1)
