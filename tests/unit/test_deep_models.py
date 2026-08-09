"""Unit tests for deep learning architectures and model summaries (Phase 7)."""

import torch

from eeg_mi.models.factory import count_parameters, create_model, get_model_summary


def test_model_summary_and_parameter_counter() -> None:
    """Test model parameter counter and summary output string."""
    model = create_model("cnn_lstm", num_channels=64, num_classes=4)
    total, trainable = count_parameters(model)
    summary_str = get_model_summary(model)

    assert total > 0
    assert trainable > 0
    assert "CNNLSTMModel" in summary_str
    assert "Total Parameters" in summary_str


def test_cnn_lstm_configurable_params(dummy_eeg_tensor: torch.Tensor) -> None:
    """Test CNN-LSTM accepts configurable channels, dropout, and hidden size."""
    model = create_model(
        "cnn_lstm",
        num_channels=64,
        num_classes=2,
        lstm_hidden_size=64,
        dropout=0.3,
    )
    out = model(dummy_eeg_tensor)
    assert out.shape == (4, 2)
