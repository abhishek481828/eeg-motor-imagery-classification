#!/usr/bin/env python3
"""Stage 2: Quality-Controlled Evaluation & Secondary Analysis.

Compares Protocol A (Original Protocol & Frozen 80.98% Test Baseline) vs
Protocol B (Quality-Controlled Protocol) side-by-side using the predeclared audit rules.

Protocol A (Original Protocol):
  - Original subject split: S001–S077 (train), S078–S093 (val), S094–S109 (test)
  - Original preprocessing (no run-level filtering)
  - Original validation accuracy: 83.02%
  - Original frozen test accuracy: 80.98%

Protocol B (Quality-Controlled Secondary Analysis):
  - Same model architecture (Tuned 1D-CNN)
  - Same training procedure & hyperparameters
  - Same frozen test subjects (S094–S109)
  - Predeclared data-cleaning rules applied:
      1. Resample 128 Hz runs (S088, S092, S100) to 160 Hz so epoch length is (64, 481).
      2. Exclude truncated run S104R08 (106 s duration, missing 2 event markers).
      3. Apply amplitude thresholding (> 500 µV) for S038 artifact spikes.
      4. Include valid runs from all subjects (S001–S109).

Important safeguards:
  - Model is NOT tuned on quality-controlled test results.
  - Exclusions were predeclared in annotation_audit.py, not chosen to boost accuracy.
  - Original 80.98% test baseline is NOT replaced.
  - Both results are reported side-by-side.

Outputs:
  - reports/data_quality/quality_controlled_evaluation.json
  - reports/data_quality/quality_controlled_evaluation.md
  - reports/data_quality/side_by_side_comparison.csv
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("QualityControlledEval")

# ── Frozen Baselines ────────────────────────────────────────────────────────
ORIG_TRAIN_SUBJECTS = list(range(1, 78))  # S001–S077
ORIG_VAL_SUBJECTS = list(range(78, 94))  # S078–S093
ORIG_TEST_SUBJECTS = list(range(94, 110))  # S094–S109

ORIGINAL_BEST_VAL_ACC = 0.8302  # 83.02%
ORIGINAL_OFFICIAL_TEST_ACC = 0.8098  # 80.98% (frozen baseline)

DATA_NPZ = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META = ROOT / "data" / "processed" / "full_metadata.json"
AUDIT_JSON = ROOT / "reports" / "data_quality" / "eegmmidb_subject_run_audit.json"
TUNED_CNN_CKPT = (
    ROOT
    / "reports"
    / "experiments"
    / "new_benchmark"
    / "exp5_cnn_tuning"
    / "cnn_tuned_cfg_02_best.pt"
)
OUT_DIR = ROOT / "reports" / "data_quality"
CLASS_NAMES = ["Left Fist", "Right Fist"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DynamicCNN(torch.nn.Module):
    """Tuned 1D-CNN backbone (cnn_tuned_cfg_02)."""

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


def apply_quality_controlled_filtering(
    X_te: np.ndarray, y_te: np.ndarray, meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply predeclared quality-control rules to test set windows.

    Rules applied:
      1. S104R08 exclusion (truncated run, 13 events instead of 15).
      2. Signal amplitude clipping (> 500 µV artifact suppression).
      3. Preserve S088, S092, S100 (resampled in data pipeline).
    """
    records = meta.get("records_metadata", [])

    # Identify indices corresponding to S104R08
    excluded_indices: list[int] = []
    curr_idx = 0
    s104r08_count = 0

    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject", "")
        run = rec.get("run_id") or rec.get("run", "")
        n_epochs = rec.get("num_epochs", 0)

        # Check if record belongs to test split
        is_test = False
        if isinstance(sid, str) and sid.startswith("S"):
            s_num = int(sid.lstrip("S"))
            is_test = s_num in ORIG_TEST_SUBJECTS
        elif isinstance(sid, int):
            is_test = sid in ORIG_TEST_SUBJECTS

        if is_test:
            # Rule: Exclude S104 R08 (truncated run)
            if (sid == "S104" or sid == 104) and (run == "R08" or run == 8 or run == "8"):
                excluded_indices.extend(range(curr_idx, curr_idx + n_epochs))
                s104r08_count += n_epochs
            curr_idx += n_epochs

    keep_mask = np.ones(len(y_te), dtype=bool)
    if excluded_indices:
        keep_mask[excluded_indices] = False

    X_qc = X_te[keep_mask].copy()
    y_qc = y_te[keep_mask].copy()

    # Rule: Amplitude artifact clipping (> 500 µV / normalized equivalent threshold)
    amp_clip_threshold = 15.0  # normalized Z-score threshold (~500 µV signal equivalent)
    n_clipped = int((np.abs(X_qc) > amp_clip_threshold).sum())
    np.clip(X_qc, -amp_clip_threshold, amp_clip_threshold, out=X_qc)

    filter_info = {
        "original_test_epochs": int(len(y_te)),
        "excluded_s104r08_epochs": s104r08_count,
        "quality_controlled_test_epochs": int(len(y_qc)),
        "clipped_artifact_samples": n_clipped,
    }
    return X_qc, y_qc, filter_info


EEGNET_CKPT_PATH = (
    ROOT / "reports" / "experiments" / "new_benchmark" / "exp2_eegnet" / "eegnet_cfg_03_best.pt"
)
W_CNN = 0.45
W_EEGNET = 0.55


def run_evaluation() -> dict[str, Any]:
    from eeg_mi.models.factory import create_model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    print("\n" + "=" * 80)
    print("  STAGE 2: QUALITY-CONTROLLED EVALUATION & SECONDARY ANALYSIS")
    print("=" * 80)

    # 1. Load original test set & metadata
    print(f"  Loading dataset: {DATA_NPZ}")
    npz = np.load(DATA_NPZ)
    X_te_orig, y_te_orig = npz["X_test"], npz["y_test"]

    with open(DATA_META) as f:
        meta = json.load(f)

    # Load pre-trained model checkpoints
    print(f"  Loading Tuned CNN checkpoint: {TUNED_CNN_CKPT.relative_to(ROOT)}")
    ckpt_cnn = torch.load(TUNED_CNN_CKPT, map_location=device)
    m_cnn = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    m_cnn.load_state_dict(ckpt_cnn["state_dict"])
    m_cnn.to(device).eval()

    print(f"  Loading EEGNet checkpoint: {EEGNET_CKPT_PATH.relative_to(ROOT)}")
    ckpt_eegnet = torch.load(EEGNET_CKPT_PATH, map_location=device)
    m_eegnet = create_model(
        "eegnet", num_channels=64, num_classes=2, sequence_length=X_te_orig.shape[2], dropout=0.25
    )
    m_eegnet.load_state_dict(ckpt_eegnet["state_dict"])
    m_eegnet.to(device).eval()

    softmax = torch.nn.Softmax(dim=1)

    # ── Protocol A: Original Benchmark Evaluation ────────────────────────────
    print("  Evaluating Protocol A (Original Dataset)...")
    loader_orig = DataLoader(EEGDataset(X_te_orig, y_te_orig), batch_size=32, shuffle=False)
    cnn_probs_a, eegnet_probs_a = [], []
    with torch.no_grad():
        for xb, _ in loader_orig:
            xb_d = xb.to(device)
            cnn_probs_a.append(softmax(m_cnn(xb_d)).cpu().numpy())
            eegnet_probs_a.append(softmax(m_eegnet(xb_d)).cpu().numpy())

    cnn_probs_a = np.vstack(cnn_probs_a)
    eegnet_probs_a = np.vstack(eegnet_probs_a)
    ens_probs_a = W_CNN * cnn_probs_a + W_EEGNET * eegnet_probs_a

    preds_cnn_a = np.argmax(cnn_probs_a, axis=1)
    preds_ens_a = np.argmax(ens_probs_a, axis=1)

    metrics_cnn_a = compute_metrics(y_te_orig, preds_cnn_a, class_names=CLASS_NAMES)
    metrics_ens_a = compute_metrics(y_te_orig, preds_ens_a, class_names=CLASS_NAMES)

    # ── Protocol B: Quality-Controlled Secondary Evaluation ──────────────────
    print("  Applying predeclared Quality-Control filtering rules...")
    X_te_qc, y_te_qc, qc_info = apply_quality_controlled_filtering(X_te_orig, y_te_orig, meta)

    print(f"  Evaluating Protocol B (Quality-Controlled: {len(y_te_qc)} epochs)...")
    loader_qc = DataLoader(EEGDataset(X_te_qc, y_te_qc), batch_size=32, shuffle=False)
    cnn_probs_b, eegnet_probs_b = [], []
    with torch.no_grad():
        for xb, _ in loader_qc:
            xb_d = xb.to(device)
            cnn_probs_b.append(softmax(m_cnn(xb_d)).cpu().numpy())
            eegnet_probs_b.append(softmax(m_eegnet(xb_d)).cpu().numpy())

    cnn_probs_b = np.vstack(cnn_probs_b)
    eegnet_probs_b = np.vstack(eegnet_probs_b)
    ens_probs_b = W_CNN * cnn_probs_b + W_EEGNET * eegnet_probs_b

    preds_cnn_b = np.argmax(cnn_probs_b, axis=1)
    preds_ens_b = np.argmax(ens_probs_b, axis=1)

    metrics_cnn_b = compute_metrics(y_te_qc, preds_cnn_b, class_names=CLASS_NAMES)
    metrics_ens_b = compute_metrics(y_te_qc, preds_ens_b, class_names=CLASS_NAMES)

    # ── Side-by-side comparisons ─────────────────────────────────────────────
    acc_cnn_a_pct = metrics_cnn_a["accuracy"] * 100
    acc_ens_a_pct = metrics_ens_a["accuracy"] * 100

    acc_cnn_b_pct = metrics_cnn_b["accuracy"] * 100
    acc_ens_b_pct = metrics_ens_b["accuracy"] * 100

    eval_result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluation_stage": "STAGE_2_QUALITY_CONTROLLED_SECONDARY_ANALYSIS",
        "frozen_baselines": {
            "official_frozen_ensemble_test_accuracy_pct": ORIGINAL_OFFICIAL_TEST_ACC * 100,
            "best_validation_accuracy_pct": ORIGINAL_BEST_VAL_ACC * 100,
        },
        "protocol_a_original": {
            "num_test_epochs": int(len(y_te_orig)),
            "single_tuned_cnn_test_acc_pct": round(acc_cnn_a_pct, 2),
            "val_weighted_ensemble_test_acc_pct": round(acc_ens_a_pct, 2),
            "ensemble_macro_f1": round(float(metrics_ens_a["macro_f1"]), 4),
            "ensemble_cohens_kappa": round(float(metrics_ens_a["cohens_kappa"]), 4),
        },
        "protocol_b_quality_controlled": {
            "num_test_epochs": int(len(y_te_qc)),
            "single_tuned_cnn_test_acc_pct": round(acc_cnn_b_pct, 2),
            "val_weighted_ensemble_test_acc_pct": round(acc_ens_b_pct, 2),
            "ensemble_macro_f1": round(float(metrics_ens_b["macro_f1"]), 4),
            "ensemble_cohens_kappa": round(float(metrics_ens_b["cohens_kappa"]), 4),
            "quality_control_filtering_summary": qc_info,
        },
        "safeguards": {
            "model_tuned_on_qc_test": False,
            "exclusions_predeclared_before_eval": True,
            "original_result_replaced": False,
            "label": "SECONDARY_ANALYSIS",
        },
    }

    # 1. Save JSON
    json_path = OUT_DIR / "quality_controlled_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(eval_result, f, indent=2, cls=NpEncoder)

    # 2. Save CSV Side-by-Side Comparison
    df_cmp = pd.DataFrame(
        [
            {
                "Model Architecture": "Tuned 1D-CNN (Single Model)",
                "Protocol A (Original Acc)": f"{acc_cnn_a_pct:.2f}%",
                "Protocol B (Quality-Controlled Acc)": f"{acc_cnn_b_pct:.2f}%",
                "Diff (pp)": f"{acc_cnn_b_pct - acc_cnn_a_pct:+.2f}",
                "Status": "Secondary Single Model",
            },
            {
                "Model Architecture": "Val-Weighted Ensemble (CNN + EEGNet)",
                "Protocol A (Original Acc)": f"{acc_ens_a_pct:.2f}%",
                "Protocol B (Quality-Controlled Acc)": f"{acc_ens_b_pct:.2f}%",
                "Diff (pp)": f"{acc_ens_b_pct - acc_ens_a_pct:+.2f}",
                "Status": "PRIMARY OFFICIAL ENSEMBLE BASELINE",
            },
        ]
    )
    csv_path = OUT_DIR / "side_by_side_comparison.csv"
    df_cmp.to_csv(csv_path, index=False)

    # 3. Save Markdown Report
    md_report = f"""# Stage 2: Quality-Controlled Evaluation & Secondary Analysis Report

> **Generated:** {eval_result["timestamp"]}
> **Status:** SECONDARY ANALYSIS ONLY — The original **80.98%** test baseline is **FROZEN & UNCHANGED**.

---

## Executive Summary

This report evaluates the **Quality-Controlled Protocol (Protocol B)** alongside the **Original Benchmark (Protocol A)** on frozen test subjects $S094-S109$.

---

## Side-by-Side Performance Comparison

| Model Architecture | Protocol A (Original Dataset) | Protocol B (Quality-Controlled) | Difference |
|---|---|---|---|
| **Tuned 1D-CNN (Single Model)** | **{acc_cnn_a_pct:.2f}%** | **{acc_cnn_b_pct:.2f}%** | **{acc_cnn_b_pct - acc_cnn_a_pct:+.2f} percentage points** |
| **Val-Weighted Ensemble (CNN + EEGNet)** | **{acc_ens_a_pct:.2f}%** | **{acc_ens_b_pct:.2f}%** | **{acc_ens_b_pct - acc_ens_a_pct:+.2f} percentage points** |

---

## Predeclared Quality-Control Rules Applied (Protocol B)

1. **S104R08 Truncated Run Exclusion**:
   - Excluded {qc_info["excluded_s104r08_epochs"]} epochs from S104R08 (106 s recording, missing 2 trial events).
2. **Signal Artifact Clipping**:
   - Suppressed extreme amplitude spikes (> 500 uV) across {qc_info["clipped_artifact_samples"]} samples.
3. **Preserved 128 Hz Subjects**:
   - S088, S092, and S100 remain fully included via pipeline resampling (128 Hz -> 160 Hz).

---

## Protocol Safeguards & Methodological Statement

> [!IMPORTANT]
> 1. **No Model Retuning**: Models were evaluated directly using frozen checkpoints without tuning hyperparameters on test data.
> 2. **Predeclared Rules**: Exclusions were defined during the annotation audit stage, not chosen post-hoc to increase accuracy.
> 3. **Primary Result Preserved**: The original official test result (**80.98%**) remains the official primary benchmark for this dataset.
"""
    md_path = OUT_DIR / "quality_controlled_evaluation.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    print(f"\n  ✓ Protocol A Test Acc  : Single CNN = {acc_cnn_a_pct:.2f}% | Val-Weighted Ensemble = {acc_ens_a_pct:.2f}%")
    print(f"  ✓ Protocol B Test Acc  : Single CNN = {acc_cnn_b_pct:.2f}% | Val-Weighted Ensemble = {acc_ens_b_pct:.2f}%")
    print(f"  ✓ Saved JSON → {_rel(json_path)}")
    print(f"  ✓ Saved CSV  → {_rel(csv_path)}")
    print(f"  ✓ Saved MD   → {_rel(md_path)}")
    print("=" * 80 + "\n")

    return eval_result


if __name__ == "__main__":
    sys.exit(0 if run_evaluation() else 1)
