"""Model Factory for instantiating architecture choices by name and model summaries."""

import torch.nn as nn

from eeg_mi.models.cnn import EEGCNN
from eeg_mi.models.cnn_lstm import CNNLSTMModel
from eeg_mi.models.lstm import EEGLSTM
from eeg_mi.utils.logging import get_logger

logger = get_logger("ModelFactory")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total parameter count and trainable parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model: nn.Module) -> str:
    """Format human-readable model architecture summary and parameter counts."""
    total, trainable = count_parameters(model)
    summary_lines = [
        "=" * 60,
        f"Model Architecture: {model.__class__.__name__}",
        "=" * 60,
        str(model),
        "-" * 60,
        f"Total Parameters     : {total:,}",
        f"Trainable Parameters : {trainable:,}",
        "=" * 60,
    ]
    return "\n".join(summary_lines)


def create_model(
    model_type: str,
    num_channels: int = 64,
    num_classes: int = 4,
    sequence_length: int = 480,
    **kwargs: any,
) -> nn.Module:
    """Instantiate and return requested model architecture with summary logging."""
    model_type_clean = model_type.lower()
    if model_type_clean == "cnn":
        model = EEGCNN(in_channels=num_channels, num_classes=num_classes)
    elif model_type_clean == "lstm":
        model = EEGLSTM(in_channels=num_channels, num_classes=num_classes)
    elif model_type_clean in ["cnn_lstm", "gan_cnn_lstm"]:
        model = CNNLSTMModel(
            in_channels=num_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(
        f"Instantiated {model_type} model. {count_parameters(model)[0]:,} total parameters."
    )
    return model
