#!/usr/bin/env python3
"""Interactive Motor Imagery Prediction Demo.

Allows you to select any subject (S001-S109) and trial window from the dataset,
run the trained Val-Weighted Ensemble model, and view the live prediction result:
  - Class 0: Left Fist Motor Imagery
  - Class 1: Right Fist Motor Imagery
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_mi.models.factory import create_model
from eeg_mi.utils.device import get_device

DATA_NPZ = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META = ROOT / "data" / "processed" / "full_metadata.json"
CNN_CKPT = (
    ROOT
    / "reports"
    / "experiments"
    / "new_benchmark"
    / "exp5_cnn_tuning"
    / "cnn_tuned_cfg_02_best.pt"
)
EEGNET_CKPT = (
    ROOT / "reports" / "experiments" / "new_benchmark" / "exp2_eegnet" / "eegnet_cfg_03_best.pt"
)

CLASS_NAMES = {0: "LEFT FIST IMAGERY", 1: "RIGHT FIST IMAGERY"}


class DynamicCNN(torch.nn.Module):
    def __init__(self, in_ch=64, filters=None, k_sz=15, drop=0.25, num_cls=2):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]
        layers = []
        c_in = in_ch
        for c_out in filters:
            layers.extend(
                [
                    torch.nn.Conv1d(c_in, c_out, kernel_size=k_sz, padding=k_sz // 2),
                    torch.nn.BatchNorm1d(c_out),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    torch.nn.Dropout(drop),
                ]
            )
            c_in = c_out
        self.features = torch.nn.Sequential(*layers)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(16)
        self.fc = torch.nn.Linear(filters[-1] * 16, num_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def run_demo() -> None:
    device = get_device("auto")
    print("\n" + "=" * 70)
    print("  INTERACTIVE EEG MOTOR IMAGERY PREDICTION DEMO")
    print("=" * 70)

    if not DATA_NPZ.exists():
        print(f"Error: Processed dataset not found at {DATA_NPZ}")
        return

    npz = np.load(DATA_NPZ)
    X_te, y_te = npz["X_test"], npz["y_test"]

    with open(DATA_META) as f:
        meta = json.load(f)

    test_subs = meta["subject_splits"]["test"]
    print(f"Loaded {len(y_te)} test trial windows across subjects S094-S109.")

    # Load Ensemble Models
    m_cnn = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    ckpt_cnn = torch.load(CNN_CKPT, map_location=device)
    m_cnn.load_state_dict(ckpt_cnn["state_dict"])
    m_cnn.to(device).eval()

    m_eegnet = create_model(
        "eegnet", num_channels=64, num_classes=2, sequence_length=X_te.shape[2], dropout=0.25
    )
    ckpt_eegnet = torch.load(EEGNET_CKPT, map_location=device)
    m_eegnet.load_state_dict(ckpt_eegnet["state_dict"])
    m_eegnet.to(device).eval()

    softmax = torch.nn.Softmax(dim=1)

    print("\n--- Model Ready ---")
    print(f"Val-Weighted Ensemble (Tuned 1D-CNN + EEGNet, 80.98% Test Accuracy)")

    # Sample random trial from test set
    idx = np.random.randint(0, len(y_te))
    sample = torch.tensor(X_te[idx], dtype=torch.float32).unsqueeze(0).to(device)
    true_label = y_te[idx]

    with torch.no_grad():
        p_cnn = softmax(m_cnn(sample)).cpu().numpy()[0]
        p_eegnet = softmax(m_eegnet(sample)).cpu().numpy()[0]

    p_ens = 0.45 * p_cnn + 0.55 * p_eegnet
    pred_label = int(np.argmax(p_ens))
    confidence = p_ens[pred_label] * 100

    print("\n" + "-" * 70)
    print(f"  Trial Index         : #{idx}")
    print(f"  True Label          : {CLASS_NAMES[true_label]} (Class {true_label})")
    print(f"  Model Prediction    : {CLASS_NAMES[pred_label]} (Class {pred_label})")
    print(f"  Confidence          : {confidence:.2f}%")
    print(f"  Result Match        : {'✅ MATCH (CORRECT PREDICTION)' if true_label == pred_label else '❌ MISMATCH'}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
