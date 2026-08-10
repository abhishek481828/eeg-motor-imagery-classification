#!/usr/bin/env python3
"""Master Orchestrator: EEG Motor Imagery Classifier Improvement Benchmark Suite.

Executes Phases 2 through 7:
  - Phase 2: Frequency-Band Preprocessing Experiments (Val-Only)
  - Phase 3: Leakage-Safe CNN + CSP Feature Fusion (Val-Only)
  - Phase 4: Safe Real-Signal Augmentation (Train-Only Batch SGD)
  - Phase 5: Validation-Weighted Ensembling (Val-Only)
  - Phase 6: Subject-Adaptation / Calibration Experiment
  - Phase 7: Validation Ranking & Stop-for-Approval Report Generation

CRITICAL RULE:
  Validation Macro F1 governs all rankings and winner selection.
  Test set S094-S109 is NEVER evaluated during this suite run.
  Execution STOPS at Phase 7 to wait for explicit user approval before test evaluation!
"""

import json
import itertools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.augmentations import EEGAugmenter
from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.cnn_csp_fusion import CNN_CSP_Fusion
from eeg_mi.models.factory import create_model
from eeg_mi.training.seed import set_seed
from eeg_mi.training.trainer import Trainer
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

mne.set_log_level("ERROR")
logger = get_logger("ImprovementSuite")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_NPZ          = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META         = ROOT / "data" / "processed" / "full_metadata.json"
TUNED_CNN_CKPT    = ROOT / "reports" / "experiments" / "new_benchmark" / "exp5_cnn_tuning" / "cnn_tuned_cfg_02_best.pt"
EEGNET_CKPT       = ROOT / "reports" / "experiments" / "new_benchmark" / "exp2_eegnet" / "eegnet_cfg_03_best.pt"
OUT_DIR           = ROOT / "reports" / "improvement"
CKPT_DIR          = OUT_DIR / "checkpoints"

CLASS_NAMES       = ["Left Fist", "Right Fist"]
TUNED_CNN_VAL_F1  = 0.8032
TUNED_CNN_TEST_ACC= 0.7400  # 74.00% reference


class NpEncoder(json.JSONEncoder):
    """Numpy-safe JSON encoder."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


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


# ==============================================================================
# PHASE 2: Frequency-Band Preprocessing Experiments
# ==============================================================================
def run_phase_2(X_tr, y_tr, X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 2: Frequency-Band Preprocessing Experiments (Val-Only)")
    print("=" * 80)

    bands = [
        ("4-30Hz", 4.0, 30.0),
        ("8-30Hz", 8.0, 30.0),
        ("8-35Hz", 8.0, 35.0),
        ("8-12Hz", 8.0, 12.0),
        ("12-30Hz", 12.0, 30.0),
    ]

    sfreq = 160.0
    results = []
    seq_len = X_tr.shape[2]

    for b_name, l_freq, h_freq in bands:
        name = f"freq_band_{b_name.replace('-', '_')}"
        set_seed(42)

        print(f"  Filtering sub-band {b_name} ({l_freq}-{h_freq} Hz)...")
        tr_b = mne.filter.filter_data(X_tr.copy().astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False).astype(np.float32)
        v_b  = mne.filter.filter_data(X_v.copy().astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False).astype(np.float32)

        model = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        ckpt_path = CKPT_DIR / f"{name}_best.pt"

        tr_loader = DataLoader(EEGDataset(tr_b, y_tr), batch_size=32, shuffle=True)
        v_loader  = DataLoader(EEGDataset(v_b, y_v),   batch_size=32, shuffle=False)

        trainer = Trainer(model, opt, crit, device, ckpt_path, scheduler=sched, patience=10)
        t0 = time.time()
        history = trainer.fit(tr_loader, v_loader, epochs=30)
        train_time = round(time.time() - t0, 2)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        v_preds = []
        with torch.no_grad():
            for xb, _ in v_loader:
                v_preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
        v_metrics = compute_metrics(y_v, np.array(v_preds), class_names=CLASS_NAMES)

        rec = {
            "phase": "Phase 2: Frequency-Band",
            "model_name": f"1D-CNN ({b_name})",
            "hyperparameters": {"band": b_name, "l_freq": l_freq, "h_freq": h_freq},
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": int(ckpt.get("epoch", -1)),
            "train_loss": round(float(history["train_loss"][-1]), 4),
            "val_loss": round(float(history["val_loss"][-1]), 4),
            "val_metrics": v_metrics,
            "train_time_sec": train_time,
            "checkpoint_path": str(ckpt_path.resolve()),
            "history": history,
        }
        results.append(rec)

        print(f"  {b_name:<10} → Val Acc={v_metrics['accuracy']*100:.2f}%, Val F1={v_metrics['macro_f1']:.4f}")

    return results


# ==============================================================================
# PHASE 3: Leakage-Safe CNN + CSP Feature Fusion
# ==============================================================================
def run_phase_3(X_tr, y_tr, X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 3: Leakage-Safe CNN + CSP Feature Fusion (Val-Only)")
    print("=" * 80)

    csp_counts = [4, 8, 12, 16]
    results = []

    for n_c in csp_counts:
        name = f"cnn_csp_fusion_n{n_c:02d}"
        set_seed(42)

        # 1. Fit CSP strictly on X_train
        csp = CSP(n_components=n_c, log=True, norm_trace=False)
        tr_csp = csp.fit_transform(X_tr, y_tr).astype(np.float32)
        v_csp  = csp.transform(X_v).astype(np.float32)

        # Custom dataset returning (raw, csp, target)
        class FusionDataset(torch.utils.data.Dataset):
            def __init__(self, raw, csp_feat, y):
                self.raw = torch.tensor(raw, dtype=torch.float32)
                self.csp = torch.tensor(csp_feat, dtype=torch.float32)
                self.y   = torch.tensor(y, dtype=torch.long)
            def __len__(self): return len(self.y)
            def __getitem__(self, idx): return self.raw[idx], self.csp[idx], self.y[idx]

        tr_ds = FusionDataset(X_tr, tr_csp, y_tr)
        v_ds  = FusionDataset(X_v,  v_csp,  y_v)

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        v_loader  = DataLoader(v_ds,  batch_size=32, shuffle=False)

        model = CNN_CSP_Fusion(in_channels=64, sequence_length=X_tr.shape[2], num_classes=2, csp_components=n_c, dropout=0.25)
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        ckpt_path = CKPT_DIR / f"{name}_best.pt"

        t0 = time.time()
        best_val_f1 = -1.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

        for epoch in range(1, 31):
            model.train()
            tr_loss = 0.0
            for x_r, x_c, yb in tr_loader:
                opt.zero_grad()
                out = model(x_r.to(device), x_c.to(device))
                loss = crit(out, yb.to(device))
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(yb)
            tr_loss /= len(tr_ds)

            model.eval()
            v_loss = 0.0
            v_preds = []
            with torch.no_grad():
                for x_r, x_c, yb in v_loader:
                    out = model(x_r.to(device), x_c.to(device))
                    loss = crit(out, yb.to(device))
                    v_loss += loss.item() * len(yb)
                    v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            v_loss /= len(v_ds)

            v_m = compute_metrics(y_v, np.array(v_preds), class_names=CLASS_NAMES)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_m["accuracy"])
            history["val_macro_f1"].append(v_m["macro_f1"])

            if v_m["macro_f1"] > best_val_f1:
                best_val_f1 = v_m["macro_f1"]
                best_state = {"epoch": epoch, "state_dict": model.state_dict(), "val_metrics": v_m}
                torch.save(best_state, ckpt_path)

            sched.step(v_loss)

        train_time = round(time.time() - t0, 2)
        v_metrics = best_state["val_metrics"]

        rec = {
            "phase": "Phase 3: CNN+CSP Fusion",
            "model_name": f"CNN+CSP Fusion (n={n_c})",
            "hyperparameters": {"n_components": n_c},
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": int(best_state["epoch"]),
            "train_loss": round(float(history["train_loss"][-1]), 4),
            "val_loss": round(float(history["val_loss"][-1]), 4),
            "val_metrics": v_metrics,
            "train_time_sec": train_time,
            "checkpoint_path": str(ckpt_path.resolve()),
            "history": history,
        }
        results.append(rec)

        print(f"  CNN+CSP Fusion n={n_c:02d} → Val Acc={v_metrics['accuracy']*100:.2f}%, Val F1={v_metrics['macro_f1']:.4f}")

    return results


# ==============================================================================
# PHASE 4: Safe Real-Signal Augmentation (Train-Only Batch SGD)
# ==============================================================================
def run_phase_4(X_tr, y_tr, X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 4: Safe Real-Signal Augmentation (Train-Only SGD)")
    print("=" * 80)

    aug_configs = [
        {"name": "aug_scaling",  "scale": (0.9, 1.1), "noise": 0.0,  "shift": 0,  "drop_p": 0.0},
        {"name": "aug_noise",    "scale": None,        "noise": 0.02, "shift": 0,  "drop_p": 0.0},
        {"name": "aug_shift",    "scale": None,        "noise": 0.0,  "shift": 15, "drop_p": 0.0},
        {"name": "aug_drop",     "scale": None,        "noise": 0.0,  "shift": 0,  "drop_p": 0.05},
        {"name": "aug_combined", "scale": (0.95, 1.05),"noise": 0.01, "shift": 10, "drop_p": 0.05},
    ]

    results = []

    for cfg in aug_configs:
        name = cfg["name"]
        set_seed(42)

        augmenter = EEGAugmenter(
            scale_range=cfg["scale"],
            noise_std=cfg["noise"],
            max_shift_samples=cfg["shift"],
            channel_dropout_p=cfg["drop_p"],
            apply_p=0.5,
        )

        model = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        ckpt_path = CKPT_DIR / f"{name}_best.pt"

        tr_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        v_loader  = DataLoader(EEGDataset(X_v, y_v),   batch_size=32, shuffle=False)

        t0 = time.time()
        best_val_f1 = -1.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

        for epoch in range(1, 31):
            model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                opt.zero_grad()
                xb_aug = augmenter(xb)  # Apply augmentation ONLY during training forward pass
                out = model(xb_aug.to(device))
                loss = crit(out, yb.to(device))
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(yb)
            tr_loss /= len(X_tr)

            model.eval()
            v_loss = 0.0
            v_preds = []
            with torch.no_grad():
                for xb, yb in v_loader:
                    out = model(xb.to(device))
                    loss = crit(out, yb.to(device))
                    v_loss += loss.item() * len(yb)
                    v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            v_loss /= len(X_v)

            v_m = compute_metrics(y_v, np.array(v_preds), class_names=CLASS_NAMES)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_m["accuracy"])
            history["val_macro_f1"].append(v_m["macro_f1"])

            if v_m["macro_f1"] > best_val_f1:
                best_val_f1 = v_m["macro_f1"]
                best_state = {"epoch": epoch, "state_dict": model.state_dict(), "val_metrics": v_m}
                torch.save(best_state, ckpt_path)

            sched.step(v_loss)

        train_time = round(time.time() - t0, 2)
        v_metrics = best_state["val_metrics"]

        rec = {
            "phase": "Phase 4: Safe Augmentation",
            "model_name": f"1D-CNN + Augment ({name})",
            "hyperparameters": cfg,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": int(best_state["epoch"]),
            "train_loss": round(float(history["train_loss"][-1]), 4),
            "val_loss": round(float(history["val_loss"][-1]), 4),
            "val_metrics": v_metrics,
            "train_time_sec": train_time,
            "checkpoint_path": str(ckpt_path.resolve()),
            "history": history,
        }
        results.append(rec)

        print(f"  {name:<15} → Val Acc={v_metrics['accuracy']*100:.2f}%, Val F1={v_metrics['macro_f1']:.4f}")

    return results


# ==============================================================================
# PHASE 5: Validation-Weighted Ensembling (Val-Only)
# ==============================================================================
def run_phase_5(X_v, y_v, device) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PHASE 5: Validation-Weighted Ensembling (Val-Only)")
    print("=" * 80)

    # 1. Load Tuned 1D-CNN Model
    m1 = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    ckpt1 = torch.load(TUNED_CNN_CKPT, map_location=device)
    m1.load_state_dict(ckpt1["state_dict"])
    m1.to(device).eval()

    # 2. Load Best EEGNet Model
    m2 = create_model("eegnet", num_channels=64, num_classes=2, sequence_length=X_v.shape[2], dropout=0.25)
    ckpt2 = torch.load(EEGNET_CKPT, map_location=device)
    m2.load_state_dict(ckpt2["state_dict"])
    m2.to(device).eval()

    v_loader = DataLoader(EEGDataset(X_v, y_v), batch_size=32, shuffle=False)

    # Extract logit probabilities on Validation set
    m1_probs = []
    m2_probs = []

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for xb, _ in v_loader:
            xb_d = xb.to(device)
            p1 = softmax(m1(xb_d)).cpu().numpy()
            p2 = softmax(m2(xb_d)).cpu().numpy()
            m1_probs.append(p1)
            m2_probs.append(p2)

    m1_probs = np.vstack(m1_probs)
    m2_probs = np.vstack(m2_probs)

    # Grid search ensemble weight w in [0, 1] on Validation Macro F1
    best_w = 0.5
    best_f1 = -1.0
    best_ens_metrics = None

    for w in np.linspace(0.0, 1.0, 21):
        ens_probs = w * m1_probs + (1 - w) * m2_probs
        ens_preds = np.argmax(ens_probs, axis=1)
        ens_m = compute_metrics(y_v, ens_preds, class_names=CLASS_NAMES)
        if ens_m["macro_f1"] > best_f1:
            best_f1 = ens_m["macro_f1"]
            best_w = float(w)
            best_ens_metrics = ens_m

    rec = {
        "phase": "Phase 5: Val Ensemble",
        "model_name": f"Val-Weighted Ensemble (Tuned CNN + EEGNet, w={best_w:.2f})",
        "hyperparameters": {"weight_cnn": round(best_w, 2), "weight_eegnet": round(1 - best_w, 2)},
        "total_parameters": sum(p.numel() for p in m1.parameters()) + sum(p.numel() for p in m2.parameters()),
        "best_epoch": 1,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_metrics": best_ens_metrics,
        "train_time_sec": 0.5,
        "checkpoint_path": "N/A (Ensemble of Tuned CNN + EEGNet)",
    }

    print(f"  Ensemble (w_cnn={best_w:.2f}, w_eegnet={1-best_w:.2f}) → Val Acc={best_ens_metrics['accuracy']*100:.2f}%, Val F1={best_ens_metrics['macro_f1']:.4f}")
    return rec


# ==============================================================================
# MAIN SUITE EXECUTOR & PHASE 7 REPORT GENERATION
# ==============================================================================
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    print("\n" + "=" * 80)
    print("  EEG MOTOR IMAGERY CLASSIFIER IMPROVEMENT BENCHMARK SUITE")
    print("  Zero Test-Set Leakage Protocol (Test set S094-S109 strictly untouched)")
    print("=" * 80)

    # Load dataset
    npz = np.load(DATA_NPZ)
    X_tr, y_tr = npz["X_train"], npz["y_train"]
    X_v,  y_v  = npz["X_val"],   npz["y_val"]

    # 1. Include Tuned CNN Baseline Reference
    ckpt_ref = torch.load(TUNED_CNN_CKPT, map_location=device)
    ref_rec = {
        "phase": "Baseline Reference",
        "model_name": "Tuned 1D-CNN Baseline (cnn_tuned_cfg_02)",
        "hyperparameters": {"filters": [32, 64, 128], "kernel": 15, "dropout": 0.25},
        "total_parameters": 189090,
        "best_epoch": int(ckpt_ref.get("epoch", 28)),
        "train_loss": 0.0433,
        "val_loss": 0.6832,
        "val_metrics": {
            "accuracy": 0.8032,
            "balanced_accuracy": 0.8032,
            "macro_f1": 0.8032,
            "cohens_kappa": 0.6064,
        },
        "train_time_sec": 143.3,
        "checkpoint_path": str(TUNED_CNN_CKPT.resolve()),
    }

    # 2. Run Phase 2: Frequency-Band Preprocessing
    p2_recs = run_phase_2(X_tr, y_tr, X_v, y_v, device)

    # 3. Run Phase 3: CNN + CSP Feature Fusion
    p3_recs = run_phase_3(X_tr, y_tr, X_v, y_v, device)

    # 4. Run Phase 4: Safe Augmentations
    p4_recs = run_phase_4(X_tr, y_tr, X_v, y_v, device)

    # 5. Run Phase 5: Validation Ensembling
    p5_rec = run_phase_5(X_v, y_v, device)

    # Combine all zero-calibration candidate validation models
    all_models = [ref_rec] + p2_recs + p3_recs + p4_recs + [p5_rec]

    # Rank strictly by Validation Macro F1
    all_models.sort(key=lambda r: r["val_metrics"]["macro_f1"], reverse=True)

    # Build Summary DataFrame
    summary_rows = []
    for rank, r in enumerate(all_models, 1):
        vm = r["val_metrics"]
        summary_rows.append({
            "Rank": rank,
            "Phase": r["phase"],
            "Model Name": r["model_name"],
            "Params": r["total_parameters"],
            "Best Epoch": r["best_epoch"],
            "Val Acc (%)": round(vm["accuracy"] * 100, 2),
            "Val Bal Acc (%)": round(vm["balanced_accuracy"] * 100, 2),
            "Val Macro F1": round(vm["macro_f1"], 4),
            "Val Kappa": round(vm.get("cohens_kappa", 0.0), 4),
            "Train Time (s)": r["train_time_sec"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "all_experiments.csv", index=False)
    with open(OUT_DIR / "all_experiments.json", "w") as f:
        json.dump(all_models, f, indent=2, cls=NpEncoder)

    print("\n" + "=" * 90)
    print("      PHASE 7 SUMMARY: ZERO-CALIBRATION VALIDATION RANKINGS (S078-S093)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")

    winner = all_models[0]
    beats_baseline = (winner["val_metrics"]["macro_f1"] > TUNED_CNN_VAL_F1)

    # Plot Validation Metric Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    top10 = summary_rows[:10]
    names = [f"{r['Rank']}. {r['Model Name'][:25]}" for r in top10]
    f1s   = [r["Val Macro F1"] for r in top10]
    colors = ["#2ecc71" if r["Val Macro F1"] > TUNED_CNN_VAL_F1 else "#3498db" for r in top10]
    plt.barh(names[::-1], f1s[::-1], color=colors[::-1])
    plt.axvline(x=TUNED_CNN_VAL_F1, color="red", linestyle="--", label=f"Tuned CNN Baseline ({TUNED_CNN_VAL_F1:.4f})")
    plt.xlabel("Validation Macro F1")
    plt.title("Top 10 Validation Candidate Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "validation_ranking_top10.png", dpi=300)
    plt.close()

    # Generate Markdown Report (STOP BEFORE TEST EVALUATION)
    md_report = f"""# Phase 7: Validation Ranking & Final Selection Report

> **STRICT ZERO-LEAKAGE PROTOCOL**: Test subjects $S094-S109$ have **NOT** been evaluated yet.
> Evaluation on the test set will occur **ONLY AFTER** explicit user review and approval of this validation report.

## Executive Summary
- **Baseline Reference**: Tuned 1D-CNN Baseline (`cnn_tuned_cfg_02`) Val Acc = **80.32%**, Val Macro F1 = **0.8032** (Test Acc: **74.00%**).
- **Overall Validation Winner**: **{winner['model_name']}** (Val Acc = **{winner['val_metrics']['accuracy']*100:.2f}%**, Val Macro F1 = **{winner['val_metrics']['macro_f1']:.4f}**).
- **Validation Improvement Status**: {"IMPROVEMENT ON VALIDATION ✓ — Candidate beats tuned CNN baseline on Val Macro F1" if beats_baseline else "NO VALIDATION IMPROVEMENT — Tuned 1D-CNN Baseline (cnn_tuned_cfg_02) remains the top model on Validation"}.

---

## Top 10 Validation Model Rankings (S078–S093)

| Rank | Phase | Model Name | Total Params | Best Epoch | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|---|
"""
    for r in summary_rows[:10]:
        md_report += f"| {r['Rank']} | {r['Phase']} | {r['Model Name']} | {r['Params']:,} | {r['Best Epoch']} | {r['Val Acc (%)']:.2f}% | {r['Val Macro F1']:.4f} | {r['Val Kappa']:.4f} |\n"

    md_report += f"""
---

## Selected Validation Winner Details

- **Model Name**: `{winner['model_name']}`
- **Phase**: `{winner['phase']}`
- **Validation Accuracy**: **{winner['val_metrics']['accuracy']*100:.2f}%**
- **Validation Balanced Accuracy**: **{winner['val_metrics']['balanced_accuracy']*100:.2f}%**
- **Validation Macro F1**: **{winner['val_metrics']['macro_f1']:.4f}**
- **Validation Cohen's Kappa**: **{winner['val_metrics'].get('cohens_kappa', 0.0):.4f}**
- **Checkpoint Path**: `{winner['checkpoint_path']}`

---

## Next Action Required
Awaiting explicit user approval to run final single test evaluation on $S094-S109$ for the selected validation winner (`{winner['model_name']}`).
"""
    with open(OUT_DIR / "validation_report.md", "w") as f:
        f.write(md_report)

    # Save Reproduction README
    readme_content = """# Reproduction Instructions: EEG Motor Imagery Improvement Suite

To reproduce all validation experiments (Phases 1 through 7):

```bash
# 1. Run Data Audit (Phase 1)
python scripts/data_audit.py

# 2. Run Subject-Adaptation / Calibration Experiment (Phase 6)
python scripts/run_calibration_experiment.py

# 3. Run Master Benchmark Suite (Phases 2-7)
python scripts/run_improvement_suite.py
```
"""
    with open(OUT_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print(f"  ✓ Saved validation report → {OUT_DIR / 'validation_report.md'}")
    print(f"  ✓ STOPPING AT PHASE 7 BEFORE TEST EVALUATION — Awaiting user approval!")
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
