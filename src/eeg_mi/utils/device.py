"""Hardware device resolution utility for CPU and CUDA support."""

import torch

from eeg_mi.utils.logging import get_logger

logger = get_logger("DeviceUtils")


def get_device(preference: str = "auto") -> torch.device:
    """Return appropriate torch.device based on availability and preference.

    Automatically selects torch.device("cuda" if torch.cuda.is_available() else "cpu").
    Does not require CUDA or an NVIDIA GPU.
    """
    if preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        # Automatic selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Selected compute device: {device}")
    return device
