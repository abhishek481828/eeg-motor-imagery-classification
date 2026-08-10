"""Multi-Seed Training & Averaging Ensemble Module (Post-Final Study).

Trains 5 random seeds (42, 123, 2024, 31415, 999) for both Tuned 1D-CNN
and EEGNet models on training subjects S001-S077.
Creates seed-averaged probability ensembles and 10-model super ensembles on S078-S093.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.factory import create_model
from eeg_mi.training.seed import set_seed
from eeg_mi.training.trainer import Trainer
from eeg_mi.utils.logging import get_logger

logger = get_logger("MultiSeedTrainer")

CLASS_NAMES = ["Left Fist", "Right Fist"]
SEEDS = [42, 123, 2024, 31415, 999]


class DynamicCNN(torch.nn.Module):
    def __init__(self, in_ch=64, filters=None, k_sz=15, drop=0.25, num_cls=2):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]
        layers = []
        c_in = in_ch
        for c_out in filters:
            layers.extend([
                torch.nn.Conv1d(c_in, c_out, kernel_size=k_sz, padding=k_sz // 2),
                torch.nn.BatchNorm1d(c_out),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
                torch.nn.Dropout(drop),
            ])
            c_in = c_out
        self.features = torch.nn.Sequential(*layers)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(16)
        self.fc = torch.nn.Linear(filters[-1] * 16, num_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_multi_seed_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_v: np.ndarray,
    y_v: np.ndarray,
    device: torch.device,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """Train 5 seeds of CNN and 5 seeds of EEGNet, returning predictions and metrics."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    softmax = torch.nn.Softmax(dim=1)

    cnn_val_probs = []
    eegnet_val_probs = []
    cnn_records = []
    eegnet_records = []

    v_loader = DataLoader(EEGDataset(X_v, y_v), batch_size=32, shuffle=False)

    # 1. Train 5 seeds of 1D-CNN
    for seed in SEEDS:
        name = f"cnn_seed_{seed}"
        set_seed(seed)

        m = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
        opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        ckpt_path = ckpt_dir / f"{name}_best.pt"

        tr_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=32, shuffle=True)

        t0 = time.time()
        trainer = Trainer(m, opt, crit, device, ckpt_path, scheduler=sched, patience=10)
        history = trainer.fit(tr_loader, v_loader, epochs=25)
        t_sec = round(time.time() - t0, 2)

        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()

        probs = []
        with torch.no_grad():
            for xb, _ in v_loader:
                probs.append(softmax(m(xb.to(device))).cpu().numpy())
        probs = np.vstack(probs)
        cnn_val_probs.append(probs)

        v_preds = np.argmax(probs, axis=1)
        v_m = compute_metrics(y_v, v_preds, class_names=CLASS_NAMES)
        cnn_records.append({
            "seed": seed,
            "metrics": v_m,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "train_time_sec": t_sec,
            "ckpt_path": str(ckpt_path.resolve()),
        })

    # 2. Train 5 seeds of EEGNet
    for seed in SEEDS:
        name = f"eegnet_seed_{seed}"
        set_seed(seed)

        m = create_model("eegnet", num_channels=64, num_classes=2, sequence_length=X_tr.shape[2], dropout=0.25)
        opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        ckpt_path = ckpt_dir / f"{name}_best.pt"

        tr_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=32, shuffle=True)

        t0 = time.time()
        trainer = Trainer(m, opt, crit, device, ckpt_path, scheduler=sched, patience=10)
        history = trainer.fit(tr_loader, v_loader, epochs=25)
        t_sec = round(time.time() - t0, 2)

        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()

        probs = []
        with torch.no_grad():
            for xb, _ in v_loader:
                probs.append(softmax(m(xb.to(device))).cpu().numpy())
        probs = np.vstack(probs)
        eegnet_val_probs.append(probs)

        v_preds = np.argmax(probs, axis=1)
        v_m = compute_metrics(y_v, v_preds, class_names=CLASS_NAMES)
        eegnet_records.append({
            "seed": seed,
            "metrics": v_m,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "train_time_sec": t_sec,
            "ckpt_path": str(ckpt_path.resolve()),
        })

    # 3. Seed-Average Predictions
    avg_cnn_probs = np.mean(cnn_val_probs, axis=0)
    avg_eegnet_probs = np.mean(eegnet_val_probs, axis=0)

    # 4. Super Ensemble (CNN 5-seed avg + EEGNet 5-seed avg)
    super_probs = 0.45 * avg_cnn_probs + 0.55 * avg_eegnet_probs

    m_avg_cnn    = compute_metrics(y_v, np.argmax(avg_cnn_probs, axis=1), class_names=CLASS_NAMES)
    m_avg_eegnet = compute_metrics(y_v, np.argmax(avg_eegnet_probs, axis=1), class_names=CLASS_NAMES)
    m_super      = compute_metrics(y_v, np.argmax(super_probs, axis=1), class_names=CLASS_NAMES)

    return {
        "cnn_seed_records": cnn_records,
        "eegnet_seed_records": eegnet_records,
        "5_seed_cnn_avg_metrics": m_avg_cnn,
        "5_seed_eegnet_avg_metrics": m_avg_eegnet,
        "10_model_super_ensemble_metrics": m_super,
    }
