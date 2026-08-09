"""Model Trainer class orchestrating model training, LR scheduling, and checkpointing."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.training.callbacks import EarlyStopping, ModelCheckpoint
from eeg_mi.utils.logging import get_logger

logger = get_logger("Trainer")


class Trainer:
    """PyTorch Model Trainer with LR scheduler, early stopping, and CPU/CUDA safe execution."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_path: Path,
        scheduler: Any = None,
        patience: int = 15,
        config_dict: dict[str, Any] | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint = ModelCheckpoint(checkpoint_path)
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.config_dict = config_dict or {}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run single training epoch with automatic memory error handling."""
        self.model.train()
        total_loss = 0.0
        try:
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(y_batch)
        except (MemoryError, RuntimeError) as err:
            if "out of memory" in str(err).lower() or isinstance(err, MemoryError):
                logger.error(
                    "Memory boundary exceeded during epoch. Reduce batch_size in configuration."
                )
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            raise err

        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> tuple[float, dict[str, Any]]:
        """Evaluate model loss, accuracy, and macro F1 using configured device."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        return avg_loss, metrics

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100
    ) -> dict[str, list]:
        """Train model over specified epochs."""
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "val_balanced_acc": [],
        }

        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, v_metrics = self.evaluate(val_loader)

            v_acc = v_metrics["accuracy"]
            v_macro_f1 = v_metrics["macro_f1"]
            v_bal_acc = v_metrics["balanced_accuracy"]

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_macro_f1"].append(v_macro_f1)
            history["val_balanced_acc"].append(v_bal_acc)

            # LR Scheduler Step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

            logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                f"Val Acc: {v_acc:.4f} | Val Macro F1: {v_macro_f1:.4f}"
            )

            # Save checkpoint based on validation macro F1
            is_best = self.early_stopping(v_macro_f1)
            if is_best:
                checkpoint_meta = {
                    "val_macro_f1": v_macro_f1,
                    "val_acc": v_acc,
                    "val_balanced_acc": v_bal_acc,
                    "config": self.config_dict,
                }
                self.checkpoint.save(self.model, epoch, checkpoint_meta)

            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return history
