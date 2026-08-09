#!/usr/bin/env python3
"""Inference/Prediction script for real-time EEG signal window classification."""

import argparse

import numpy as np
import torch

from eeg_mi.models.factory import create_model
from eeg_mi.utils.logging import get_logger

logger = get_logger("PredictScript")


def predict_sample(model: torch.nn.Module, sample: np.ndarray) -> int:
    """Predict class index for an EEG window shape (C, T)."""
    model.eval()
    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time EEG window prediction CLI")
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=480)
    args = parser.parse_args()

    model = create_model("cnn_lstm", num_channels=args.num_channels, num_classes=4)
    dummy_window = np.random.randn(args.num_channels, args.seq_len)
    class_idx = predict_sample(model, dummy_window)
    logger.info(f"Predicted class index: {class_idx}")


if __name__ == "__main__":
    main()
