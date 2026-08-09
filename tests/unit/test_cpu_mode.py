"""Unit tests verifying CPU mode execution, device resolution, and CPU-safe defaults."""

import torch

from eeg_mi.models.factory import create_model
from eeg_mi.utils.device import get_device


def test_cpu_device_resolution() -> None:
    """Test get_device returns torch.device('cpu') when preference is 'cpu' or CUDA is disabled."""
    cpu_device = get_device("cpu")
    assert isinstance(cpu_device, torch.device)
    assert cpu_device.type == "cpu"

    auto_device = get_device("auto")
    assert isinstance(auto_device, torch.device)
    assert auto_device.type in ["cpu", "cuda"]


def test_model_cpu_execution() -> None:
    """Test instantiating CNN-LSTM model on CPU and running forward pass without CUDA."""
    device = torch.device("cpu")
    model = create_model("cnn_lstm", num_channels=64, num_classes=2)
    model.to(device)

    dummy_input = torch.randn(8, 64, 480, device=device)
    outputs = model(dummy_input)

    assert outputs.device.type == "cpu"
    assert outputs.shape == (8, 2)
