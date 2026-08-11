#!/usr/bin/env python3
"""Phase A: Error Analysis & Baseline Subject Profiling.

Evaluates the baseline 45% 1D-CNN + 55% EEGNet ensemble on validation subjects (S078-S093).
Generates:
- Subject-wise accuracy, Macro F1, and Cohen's Kappa
- Confusion matrix and per-class error distribution
- Prediction confidence distributions for correct vs incorrect predictions
- Hard subject identification (< 75% accuracy)
- Sub-band PSD spectral energy analysis comparing correct vs misclassified trials
- Saved JSON summary & Markdown report
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.factory import create_model
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("ErrorAnalysis")

DATA_NPZ = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META = ROOT / "data" / "processed" / "full_metadata.json"
CKPT_DIR = ROOT / "models" / "checkpoints" / "post_final_improvements"
OUT_DIR = ROOT / "reports" / "experiments" / "accuracy_improvement_study" / "phase_a_error_analysis"

CLASS_NAMES = ["Left Fist", "Right Fist"]


class NpEncoder(json.JSONEncoder):
    """Numpy-safe JSON encoder."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute per-subject performance metrics and prediction confidence statistics."""
    val_subs = meta["subject_splits"]["validation"]
    records = meta.get("records_metadata", [])

    # Calculate number of trials per subject in validation split
    sub_counts = {int(s): 0 for s in val_subs}
    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject", 0)
        if isinstance(sid, str) and sid.startswith("S"):
            sid = int(sid[1:])
        if int(sid) in sub_counts:
            sub_counts[int(sid)] += 1

    breakdown = {}
    offset = 0
    for s in val_subs:
        s_int = int(s)
        s_str = f"S{s_int:03d}"
        n_ep = sub_counts.get(s_int, 0)
        s_y = y_true[offset : offset + n_ep]
        s_p = y_pred[offset : offset + n_ep]
        s_prob = probs[offset : offset + n_ep]
        offset += n_ep

        if len(s_y) == 0:
            continue

        s_m = compute_metrics(s_y, s_p, class_names=CLASS_NAMES)
        confidences = np.max(s_prob, axis=1)
        correct_mask = s_y == s_p

        breakdown[s_str] = {
            "num_epochs": int(len(s_y)),
            "accuracy": round(float(s_m["accuracy"]), 4),
            "macro_f1": round(float(s_m["macro_f1"]), 4),
            "cohens_kappa": round(float(s_m["cohens_kappa"]), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "mean_correct_confidence": round(
                float(np.mean(confidences[correct_mask])) if np.sum(correct_mask) > 0 else 0.0, 4
            ),
            "mean_error_confidence": round(
                float(np.mean(confidences[~correct_mask])) if np.sum(~correct_mask) > 0 else 0.0, 4
            ),
            "confusion_matrix": s_m["confusion_matrix"],
        }

    return breakdown


def analyze_spectral_errors(
    X_val: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, sfreq: float = 160.0
) -> dict[str, Any]:
    """Compute sub-band spectral power differences between correct and misclassified trials."""
    correct_mask = y_true == y_pred
    error_mask = ~correct_mask

    bands = {
        "theta": (4.0, 8.0),
        "mu": (8.0, 12.0),
        "low_beta": (12.0, 18.0),
        "high_beta": (18.0, 26.0),
        "gamma": (26.0, 40.0),
    }

    freqs, psd = signal.welch(X_val, fs=sfreq, axis=-1)  # (N, C, n_freqs)
    psd_mean = np.mean(psd, axis=1)  # Average across channels -> (N, n_freqs)

    correct_psd = psd_mean[correct_mask]
    error_psd = psd_mean[error_mask]

    spectral_report = {}
    for b_name, (f_min, f_max) in bands.items():
        idx_b = np.logical_and(freqs >= f_min, freqs <= f_max)
        corr_p = float(np.mean(correct_psd[:, idx_b])) if len(correct_psd) > 0 else 0.0
        err_p = float(np.mean(error_psd[:, idx_b])) if len(error_psd) > 0 else 0.0
        ratio = float(err_p / corr_p) if corr_p > 0 else 1.0

        spectral_report[b_name] = {
            "freq_range": [f_min, f_max],
            "correct_trials_psd_power": round(corr_p, 6),
            "error_trials_psd_power": round(err_p, 6),
            "error_to_correct_power_ratio": round(ratio, 4),
        }

    return spectral_report


class DynamicCNN(torch.nn.Module):
    def __init__(
        self,
        in_ch: int = 64,
        filters: list[int] | None = None,
        k_sz: int = 15,
        drop: float = 0.25,
        num_cls: int = 2,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    logger.info("Starting Phase A: Error Analysis & Baseline Subject Profiling...")

    # Load dataset & metadata
    npz = np.load(DATA_NPZ)
    X_v, y_v = npz["X_val"], npz["y_val"]
    with open(DATA_META) as f:
        meta = json.load(f)

    seq_len = X_v.shape[2]

    # Load baseline CNN & EEGNet checkpoints (seed 42)
    cnn_model = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    cnn_ckpt = torch.load(CKPT_DIR / "cnn_seed_42_best.pt", map_location=device)
    cnn_model.load_state_dict(cnn_ckpt["state_dict"])
    cnn_model.to(device).eval()

    eegnet_model = create_model(
        "eegnet", num_channels=64, num_classes=2, sequence_length=seq_len, dropout=0.25
    )
    eegnet_ckpt = torch.load(CKPT_DIR / "eegnet_seed_42_best.pt", map_location=device)
    eegnet_model.load_state_dict(eegnet_ckpt["state_dict"])
    eegnet_model.to(device).eval()

    v_loader = DataLoader(EEGDataset(X_v, y_v), batch_size=32, shuffle=False)

    # Get predictions and softmax probabilities
    cnn_probs, eegnet_probs = [], []
    with torch.no_grad():
        for xb, _ in v_loader:
            xb = xb.to(device)
            p_cnn = torch.softmax(cnn_model(xb), dim=1).cpu().numpy()
            p_eeg = torch.softmax(eegnet_model(xb), dim=1).cpu().numpy()
            cnn_probs.append(p_cnn)
            eegnet_probs.append(p_eeg)

    cnn_probs = np.vstack(cnn_probs)
    eegnet_probs = np.vstack(eegnet_probs)

    # 45% CNN + 55% EEGNet Ensemble Probabilities
    ensemble_probs = 0.45 * cnn_probs + 0.55 * eegnet_probs
    y_pred = np.argmax(ensemble_probs, axis=1)

    # Global Validation Metrics
    val_metrics = compute_metrics(y_v, y_pred, class_names=CLASS_NAMES)
    logger.info(
        f"Baseline Ensemble Val Acc: {val_metrics['accuracy'] * 100:.2f}%, Val Macro F1: {val_metrics['macro_f1']:.4f}"
    )

    # Subject Breakdown
    sub_metrics = per_subject_metrics(y_v, y_pred, ensemble_probs, meta)

    # Identify hard subjects (< 75% accuracy)
    hard_subjects = {s: m for s, m in sub_metrics.items() if m["accuracy"] < 0.75}

    # Spectral Error Analysis
    spectral_analysis = analyze_spectral_errors(X_v, y_v, y_pred)

    # Save Error Analysis Data JSON
    report_data = {
        "model_name": "Baseline Ensemble (45% CNN + 55% EEGNet)",
        "overall_val_metrics": val_metrics,
        "per_subject_metrics": sub_metrics,
        "hard_subjects": hard_subjects,
        "spectral_error_analysis": spectral_analysis,
    }
    with open(OUT_DIR / "error_analysis_data.json", "w") as f:
        json.dump(report_data, f, indent=2, cls=NpEncoder)

    # Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = np.array(val_metrics["confusion_matrix"])
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Phase A: Baseline Ensemble Validation Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "validation_confusion_matrix.png", dpi=300)
    plt.close()

    # Generate Markdown Report
    md_lines = [
        "# Phase A: Error Analysis & Baseline Subject Profiling Report",
        "",
        "## 1. Overall Baseline Validation Performance",
        f"- **Validation Accuracy**: **{val_metrics['accuracy'] * 100:.2f}%**",
        f"- **Validation Macro F1**: **{val_metrics['macro_f1']:.4f}**",
        f"- **Cohen's Kappa**: **{val_metrics['cohens_kappa']:.4f}**",
        f"- **Left Fist Accuracy / F1**: Recall={val_metrics['per_class']['Left Fist']['recall'] * 100:.2f}%, F1={val_metrics['per_class']['Left Fist']['f1_score']:.4f}",
        f"- **Right Fist Accuracy / F1**: Recall={val_metrics['per_class']['Right Fist']['recall'] * 100:.2f}%, F1={val_metrics['per_class']['Right Fist']['f1_score']:.4f}",
        "",
        "## 2. Hard Subject Breakdown (< 75% Accuracy)",
    ]

    for s_str, m in hard_subjects.items():
        md_lines.append(
            f"- **Subject {s_str}**: Accuracy = **{m['accuracy'] * 100:.2f}%**, Macro F1 = **{m['macro_f1']:.4f}**, "
            f"Mean Error Conf = {m['mean_error_confidence']:.4f}"
        )

    md_lines.extend(
        [
            "",
            "## 3. Sub-band Spectral Power Error Analysis",
            "Ratio of mean PSD power in misclassified trials vs. correctly classified trials:",
        ]
    )

    for b_name, b_info in spectral_analysis.items():
        md_lines.append(
            f"- **{b_name.upper()} ({b_info['freq_range'][0]}-{b_info['freq_range'][1]} Hz)**: "
            f"Error/Correct Ratio = **{b_info['error_to_correct_power_ratio']:.4f}** "
            f"(Err Power: {b_info['error_trials_psd_power']:.6f}, Corr Power: {b_info['correct_trials_psd_power']:.6f})"
        )

    md_path = OUT_DIR / "PHASE_A_ERROR_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info(f"Phase A Error Analysis completed! Reports saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
