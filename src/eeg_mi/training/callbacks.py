"""Training callbacks (EarlyStopping and ModelCheckpoint)."""

from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping trigger based on validation metric improvement."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (
            (current_score > self.best_score)
            if self.mode == "max"
            else (current_score < self.best_score)
        )
        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpoint:
    """Model checkpoint saver."""

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, epoch: int, metrics: dict) -> None:
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, self.filepath)
