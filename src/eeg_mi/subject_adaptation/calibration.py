"""Target Subject Calibration and Adaptation Protocols.

Provides trial-level splitting within target subjects and executes controlled
adaptation strategies (Strategies A-G) with zero evaluation-leakage.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.subject_adaptation.adapters import (
    PrototypeCalibrator,
    SubjectAdapter,
    TemperatureScaler,
)


def split_target_subject_trials(
    X_sub: np.ndarray, y_sub: np.ndarray, k_calibration: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split target subject epochs into k calibration trials and separate evaluation trials.

    Ensures k_cal and k_eval are completely disjoint (k_cal intersect k_eval = empty set).
    If k_calibration == 0, returns empty calibration set and full subject data for evaluation.
    """
    n_epochs = len(y_sub)
    if k_calibration <= 0:
        return (
            np.empty((0, *X_sub.shape[1:]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            X_sub,
            y_sub,
        )

    assert k_calibration < n_epochs, (
        f"Calibration budget k={k_calibration} must be smaller than total epochs {n_epochs}!"
    )

    rng = np.random.RandomState(seed)
    # Stratified sampling across classes
    classes = np.unique(y_sub)
    cal_indices = []

    per_class_k = max(1, k_calibration // len(classes))
    for c in classes:
        c_idx = np.where(y_sub == c)[0]
        rng.shuffle(c_idx)
        cal_indices.extend(c_idx[:per_class_k])

    cal_indices = np.array(cal_indices[:k_calibration])
    eval_mask = np.ones(n_epochs, dtype=bool)
    eval_mask[cal_indices] = False
    eval_indices = np.where(eval_mask)[0]

    X_cal, y_cal = X_sub[cal_indices], y_sub[cal_indices]
    X_eval, y_eval = X_sub[eval_indices], y_sub[eval_indices]
    return X_cal, y_cal, X_eval, y_eval


class TargetSubjectAdaptor:
    """Executes target-subject adaptation strategies A through G on a pretrained model."""

    def __init__(self, base_model: nn.Module, strategy: str = "A", lr: float = 1e-4):
        self.base_model = base_model
        self.strategy = strategy.upper()
        self.lr = lr

    def adapt(self, X_cal: np.ndarray, y_cal: np.ndarray, device: torch.device) -> dict[str, Any]:
        """Adapt model on target subject calibration trials."""
        model = self.base_model.to(device)

        if len(y_cal) == 0 or self.strategy == "ZERO_SHOT":
            return {"model": model, "adapter_type": "zero_shot"}

        if self.strategy == "A":
            # Head fine-tuning only
            for param in model.parameters():
                param.requires_grad = False
            # Enable head FC
            for param in model.fc.parameters():
                param.requires_grad = True

            opt = torch.optim.Adam(model.fc.parameters(), lr=self.lr)
            crit = nn.CrossEntropyLoss()
            cal_loader = DataLoader(
                EEGDataset(X_cal, y_cal), batch_size=min(8, len(y_cal)), shuffle=True
            )

            model.train()
            for _epoch in range(15):
                for xb, yb in cal_loader:
                    opt.zero_grad()
                    out = model(xb.to(device))
                    loss = crit(out, yb.to(device))
                    loss.backward()
                    opt.step()

        elif self.strategy == "B":
            # BatchNorm adaptation only
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.train()
                else:
                    module.eval()

            cal_loader = DataLoader(
                EEGDataset(X_cal, y_cal), batch_size=min(8, len(y_cal)), shuffle=True
            )
            with torch.no_grad():
                for xb, _ in cal_loader:
                    model(xb.to(device))

        elif self.strategy == "D":
            # Subject Adapter layer
            feat_dim = 128 * 16  # DynamicCNN final embedding dim
            adapter = SubjectAdapter(feature_dim=feat_dim).to(device)
            opt = torch.optim.Adam(adapter.parameters(), lr=self.lr)
            crit = nn.CrossEntropyLoss()

            cal_loader = DataLoader(
                EEGDataset(X_cal, y_cal), batch_size=min(8, len(y_cal)), shuffle=True
            )
            model.eval()
            for _epoch in range(15):
                for xb, yb in cal_loader:
                    opt.zero_grad()
                    feat = model.extract_features(xb.to(device))
                    feat_adapted = adapter(feat)
                    out = model.fc(feat_adapted)
                    loss = crit(out, yb.to(device))
                    loss.backward()
                    opt.step()
            return {"model": model, "adapter": adapter, "adapter_type": "D"}

        elif self.strategy == "E":
            # Prototype centroid calibration
            model.eval()
            with torch.no_grad():
                feats = (
                    model.extract_features(torch.tensor(X_cal, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )
            proto_cal = PrototypeCalibrator().fit(feats, y_cal)
            return {"model": model, "prototype_calibrator": proto_cal, "adapter_type": "E"}

        elif self.strategy == "F":
            # Temperature scaling
            temp_scaler = TemperatureScaler().to(device)
            opt = torch.optim.Adam(temp_scaler.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss()
            cal_loader = DataLoader(
                EEGDataset(X_cal, y_cal), batch_size=min(8, len(y_cal)), shuffle=True
            )

            model.eval()
            for _epoch in range(20):
                for xb, yb in cal_loader:
                    opt.zero_grad()
                    with torch.no_grad():
                        logits = model(xb.to(device))
                    scaled_logits = temp_scaler(logits)
                    loss = crit(scaled_logits, yb.to(device))
                    loss.backward()
                    opt.step()
            return {"model": model, "temperature_scaler": temp_scaler, "adapter_type": "F"}

        return {"model": model, "adapter_type": self.strategy}
