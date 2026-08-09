"""Loss functions for PyTorch training."""

import torch.nn as nn


def get_loss_function(name: str = "cross_entropy") -> nn.Module:
    """Return requested loss criterion."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss function: {name}")
