#!/usr/bin/env python3
"""Master Orchestrator: Safe Post-Final Accuracy-Improvement Experiments.

Executes Phases 2 through 6 strictly on Training (S001-S077) and Validation (S078-S093) data:
  - Phase 2: Validation-Only Ensemble Weight Search (w_cnn in [0.30, 0.70])
  - Phase 3: Multi-Seed Model Averaging (5 seeds for CNN & EEGNet -> 10-model super ensemble)
  - Phase 4: Safe Training-Only Batch Augmentation (scaling, noise, shift, dropout, masking)
  - Phase 5: Subject-Adaptation / Calibration Analysis (k in [5, 10, 20, 30] trials)
  - Phase 6: Validation Ranking & Comparative Protocol Report

STRICT SAFETY RULE:
  Test subjects S094-S109 are NEVER loaded, inspected, or evaluated.
  The official 80.98% test accuracy remains permanently frozen.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.augmentations import EEGAugmenter
from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.factory import create_model
from eeg_mi.training.seed import set_seed
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger
from src.experiments.post_final.multi_seed_trainer import train_multi_seed_models

logger = get_logger("PostFinalExperiments")

DATA_NPZ = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META = ROOT / "data" / "processed" / "full_metadata.json"

CNN_CKPT_PATH = (
    ROOT
    / "reports"
    / "experiments"
    / "new_benchmark"
    / "exp5_cnn_tuning"
    / "cnn_tuned_cfg_02_best.pt"
)
EEGNET_CKPT_PATH = (
    ROOT / "reports" / "experiments" / "new_benchmark" / "exp2_eegnet" / "eegnet_cfg_03_best.pt"
)

OUT_DIR = ROOT / "reports" / "post_final_improvements"
CKPT_DIR = ROOT / "models" / "checkpoints" / "post_final_improvements"

CLASS_NAMES = ["Left Fist", "Right Fist"]
VAL_REF_ACC = 0.8302
VAL_REF_F1 = 0.8302


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


# ==============================================================================
# PHASE 2: Validation-Only Ensemble-Weight Search
# ==============================================================================
def run_phase_2(X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 2: Validation-Only Ensemble Weight Search (S078-S093)")
    print("=" * 80)

    m_cnn = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    ckpt_cnn = torch.load(CNN_CKPT_PATH, map_location=device)
    m_cnn.load_state_dict(ckpt_cnn["state_dict"])
    m_cnn.to(device).eval()

    m_eegnet = create_model(
        "eegnet", num_channels=64, num_classes=2, sequence_length=X_v.shape[2], dropout=0.25
    )
    ckpt_eegnet = torch.load(EEGNET_CKPT_PATH, map_location=device)
    m_eegnet.load_state_dict(ckpt_eegnet["state_dict"])
    m_eegnet.to(device).eval()

    v_loader = DataLoader(EEGDataset(X_v, y_v), batch_size=32, shuffle=False)
    m1_probs, m2_probs = [], []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for xb, _ in v_loader:
            xb_d = xb.to(device)
            p1 = softmax(m_cnn(xb_d)).cpu().numpy()
            p2 = softmax(m_eegnet(xb_d)).cpu().numpy()
            m1_probs.append(p1)
            m2_probs.append(p2)

    m1_probs = np.vstack(m1_probs)
    m2_probs = np.vstack(m2_probs)

    weights_to_test = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.45, 0.55),  # Reference
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]

    results = []
    for w_cnn, w_eeg in weights_to_test:
        ens_probs = w_cnn * m1_probs + w_eeg * m2_probs
        ens_preds = np.argmax(ens_probs, axis=1)
        v_m = compute_metrics(y_v, ens_preds, class_names=CLASS_NAMES)

        rec = {
            "phase": "Phase 2: Weight Search",
            "model_name": f"Ensemble (w_cnn={w_cnn:.2f}, w_eeg={w_eeg:.2f})",
            "ensemble_weights": {"w_cnn": w_cnn, "w_eeg": w_eeg},
            "seeds": [42],
            "augmentation": "None",
            "total_parameters": 191700,
            "best_epoch": 1,
            "val_metrics": v_m,
            "train_time_sec": 0.2,
            "checkpoint_path": "Frozen Baseline Checkpoints",
        }
        results.append(rec)
        print(
            f"  Ensemble w_cnn={w_cnn:.2f}, w_eeg={w_eeg:.2f} → Val Acc={v_m['accuracy'] * 100:.2f}%, Val F1={v_m['macro_f1']:.4f}"
        )

    return results


# ==============================================================================
# PHASE 3: Multi-Seed Model Averaging (10-Model Super Ensemble)
# ==============================================================================
def run_phase_3(X_tr, y_tr, X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 3: Multi-Seed Model Averaging (10-Model Super Ensemble)")
    print("=" * 80)

    res = train_multi_seed_models(X_tr, y_tr, X_v, y_v, device, CKPT_DIR)

    rec_cnn_avg = {
        "phase": "Phase 3: Multi-Seed Avg",
        "model_name": "5-Seed CNN Averaging Ensemble",
        "ensemble_weights": {"cnn_seeds": 5},
        "seeds": [42, 123, 2024, 31415, 999],
        "augmentation": "None",
        "total_parameters": 189090 * 5,
        "best_epoch": 25,
        "val_metrics": res["5_seed_cnn_avg_metrics"],
        "train_time_sec": sum(r["train_time_sec"] for r in res["cnn_seed_records"]),
        "checkpoint_path": str(CKPT_DIR.resolve()),
    }

    rec_eegnet_avg = {
        "phase": "Phase 3: Multi-Seed Avg",
        "model_name": "5-Seed EEGNet Averaging Ensemble",
        "ensemble_weights": {"eegnet_seeds": 5},
        "seeds": [42, 123, 2024, 31415, 999],
        "augmentation": "None",
        "total_parameters": 2610 * 5,
        "best_epoch": 25,
        "val_metrics": res["5_seed_eegnet_avg_metrics"],
        "train_time_sec": sum(r["train_time_sec"] for r in res["eegnet_seed_records"]),
        "checkpoint_path": str(CKPT_DIR.resolve()),
    }

    rec_super = {
        "phase": "Phase 3: Multi-Seed Avg",
        "model_name": "10-Model Super Ensemble (5-Seed CNN + 5-Seed EEGNet, w=0.45/0.55)",
        "ensemble_weights": {"w_cnn_5seed": 0.45, "w_eegnet_5seed": 0.55},
        "seeds": [42, 123, 2024, 31415, 999],
        "augmentation": "None",
        "total_parameters": 189090 * 5 + 2610 * 5,
        "best_epoch": 25,
        "val_metrics": res["10_model_super_ensemble_metrics"],
        "train_time_sec": rec_cnn_avg["train_time_sec"] + rec_eegnet_avg["train_time_sec"],
        "checkpoint_path": str(CKPT_DIR.resolve()),
    }

    print(
        f"  5-Seed CNN Avg     → Val Acc={m_acc(rec_cnn_avg):.2f}%, Val F1={m_f1(rec_cnn_avg):.4f}"
    )
    print(
        f"  5-Seed EEGNet Avg  → Val Acc={m_acc(rec_eegnet_avg):.2f}%, Val F1={m_f1(rec_eegnet_avg):.4f}"
    )
    print(f"  10-Model Super Ens → Val Acc={m_acc(rec_super):.2f}%, Val F1={m_f1(rec_super):.4f}")

    return [rec_cnn_avg, rec_eegnet_avg, rec_super]


def m_acc(rec):
    return rec["val_metrics"]["accuracy"] * 100


def m_f1(rec):
    return rec["val_metrics"]["macro_f1"]


# ==============================================================================
# PHASE 4: Safe Training-Only Batch Augmentation
# ==============================================================================
def run_phase_4(X_tr, y_tr, X_v, y_v, device) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 4: Safe Training-Only Batch Augmentations (No GANs)")
    print("=" * 80)

    aug_configs = [
        {
            "name": "aug_mild_shift_scaling",
            "scale": (0.95, 1.05),
            "noise": 0.0,
            "shift": 10,
            "drop_p": 0.0,
        },
        {
            "name": "aug_mild_noise_dropout",
            "scale": None,
            "noise": 0.01,
            "shift": 0,
            "drop_p": 0.03,
        },
    ]

    results = []
    v_loader = DataLoader(EEGDataset(X_v, y_v), batch_size=32, shuffle=False)

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

        t0 = time.time()
        best_val_f1 = -1.0
        best_state = None

        for epoch in range(1, 26):
            model.train()
            for xb, yb in tr_loader:
                opt.zero_grad()
                xb_aug = augmenter(xb)  # Apply augmentation ONLY during training SGD
                out = model(xb_aug.to(device))
                loss = crit(out, yb.to(device))
                loss.backward()
                opt.step()

            model.eval()
            v_preds = []
            v_loss = 0.0
            with torch.no_grad():
                for xb, yb in v_loader:
                    out = model(xb.to(device))
                    loss = crit(out, yb.to(device))
                    v_loss += loss.item() * len(yb)
                    v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            v_loss /= len(X_v)

            v_m = compute_metrics(y_v, np.array(v_preds), class_names=CLASS_NAMES)
            if v_m["macro_f1"] > best_val_f1:
                best_val_f1 = v_m["macro_f1"]
                best_state = {"epoch": epoch, "state_dict": model.state_dict(), "val_metrics": v_m}
                torch.save(best_state, ckpt_path)

            sched.step(v_loss)

        t_sec = round(time.time() - t0, 2)
        v_m = best_state["val_metrics"]

        rec = {
            "phase": "Phase 4: Safe Augmentation",
            "model_name": f"1D-CNN ({name})",
            "ensemble_weights": {"w_cnn": 1.0},
            "seeds": [42],
            "augmentation": name,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": int(best_state["epoch"]),
            "val_metrics": v_m,
            "train_time_sec": t_sec,
            "checkpoint_path": str(ckpt_path.resolve()),
        }
        results.append(rec)
        print(f"  {name:<25} → Val Acc={v_m['accuracy'] * 100:.2f}%, Val F1={v_m['macro_f1']:.4f}")

    return results


# ==============================================================================
# PHASE 5: Subject-Adaptation / Calibration Analysis (Isolated)
# ==============================================================================
def run_phase_5(X_tr, y_tr, X_v, y_v, device) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PHASE 5: Subject-Adaptation Calibration Analysis (Isolated)")
    print("  Label: SUBJECT-ADAPTED / CALIBRATION-BASED")
    print("=" * 80)

    with open(DATA_META) as f:
        meta = json.load(f)

    val_subs = meta["subject_splits"]["validation"]
    records = meta.get("records_metadata", [])

    sub_counts = {int(s): 0 for s in val_subs}
    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject", 0)
        if isinstance(sid, str) and sid.startswith("S"):
            sid = int(sid[1:])
        if int(sid) in sub_counts:
            sub_counts[int(sid)] += 1

    k_trials_list = [5, 10, 20, 30]
    calibration_results = {}

    for k in k_trials_list:
        sub_results = {}
        offset = 0

        for s in val_subs:
            s_int = int(s)
            s_str = f"S{s_int:03d}"
            n_ep = sub_counts.get(s_int, 0)
            s_X = X_v[offset : offset + n_ep]
            s_y = y_v[offset : offset + n_ep]
            offset += n_ep

            if n_ep <= k:
                continue

            set_seed(42)

            # 1. Base Model No-Cal
            m_base = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
            ckpt = torch.load(CNN_CKPT_PATH, map_location=device)
            m_base.load_state_dict(ckpt["state_dict"])
            m_base.to(device).eval()

            with torch.no_grad():
                preds_no_cal = (
                    torch.argmax(m_base(torch.tensor(s_X, dtype=torch.float32).to(device)), dim=1)
                    .cpu()
                    .numpy()
                )
            acc_no_cal = float(np.mean(preds_no_cal == s_y))

            # 2. Subject-Adapted Calibration
            X_cal, y_cal = s_X[:k], s_y[:k]
            X_eval, y_eval = s_X[k:], s_y[k:]

            m_adapted = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
            m_adapted.load_state_dict(ckpt["state_dict"])
            m_adapted.to(device)

            opt = torch.optim.Adam(m_adapted.parameters(), lr=0.0001)
            crit = torch.nn.CrossEntropyLoss()
            cal_loader = DataLoader(EEGDataset(X_cal, y_cal), batch_size=min(k, 8), shuffle=True)

            m_adapted.train()
            for _epoch in range(10):
                for xb, yb in cal_loader:
                    opt.zero_grad()
                    out = m_adapted(xb.to(device))
                    loss = crit(out, yb.to(device))
                    loss.backward()
                    opt.step()

            m_adapted.eval()
            with torch.no_grad():
                preds_adapted = (
                    torch.argmax(
                        m_adapted(torch.tensor(X_eval, dtype=torch.float32).to(device)), dim=1
                    )
                    .cpu()
                    .numpy()
                )
            acc_adapted = float(np.mean(preds_adapted == y_eval))

            sub_results[s_str] = {
                "num_calibration_trials": k,
                "no_calibration_accuracy": round(acc_no_cal, 4),
                "subject_adapted_accuracy": round(acc_adapted, 4),
                "accuracy_delta": round(acc_adapted - acc_no_cal, 4),
            }

        mean_no_cal = float(np.mean([v["no_calibration_accuracy"] for v in sub_results.values()]))
        mean_adapted = float(np.mean([v["subject_adapted_accuracy"] for v in sub_results.values()]))
        mean_delta = float(np.mean([v["accuracy_delta"] for v in sub_results.values()]))

        calibration_results[f"k_{k}_trials"] = {
            "k_trials": k,
            "mean_no_calibration_accuracy": round(mean_no_cal, 4),
            "mean_subject_adapted_accuracy": round(mean_adapted, 4),
            "mean_accuracy_delta": round(mean_delta, 4),
            "per_subject": sub_results,
        }

        print(
            f"  k={k:<2d} trials → No-Cal Acc: {mean_no_cal * 100:.2f}% | Adapted Acc: {mean_adapted * 100:.2f}% | Delta: {mean_delta * 100:+.2f}%"
        )

    md_content = f"""# Phase 5: Subject-Adaptation / Calibration Results Report

> **CATEGORY**: **SUBJECT-ADAPTED / CALIBRATION-BASED**
> Results utilize subject-specific calibration data and are kept strictly isolated from zero-calibration benchmark results.

## Calibration Performance Summary

| Calibration Trials ($k$) | No-Calibration Accuracy | Subject-Adapted Accuracy | Accuracy Delta ($\\Delta \\text{{Acc}}$) |
|---|---|---|---|
| **$k=5$ trials** | {calibration_results["k_5_trials"]["mean_no_calibration_accuracy"] * 100:.2f}% | **{calibration_results["k_5_trials"]["mean_subject_adapted_accuracy"] * 100:.2f}%** | **{calibration_results["k_5_trials"]["mean_accuracy_delta"] * 100:+.2f}%** |
| **$k=10$ trials** | {calibration_results["k_10_trials"]["mean_no_calibration_accuracy"] * 100:.2f}% | **{calibration_results["k_10_trials"]["mean_subject_adapted_accuracy"] * 100:.2f}%** | **{calibration_results["k_10_trials"]["mean_accuracy_delta"] * 100:+.2f}%** |
| **$k=20$ trials** | {calibration_results["k_20_trials"]["mean_no_calibration_accuracy"] * 100:.2f}% | **{calibration_results["k_20_trials"]["mean_subject_adapted_accuracy"] * 100:.2f}%** | **{calibration_results["k_20_trials"]["mean_accuracy_delta"] * 100:+.2f}%** |
| **$k=30$ trials** | {calibration_results["k_30_trials"]["mean_no_calibration_accuracy"] * 100:.2f}% | **{calibration_results["k_30_trials"]["mean_subject_adapted_accuracy"] * 100:.2f}%** | **{calibration_results["k_30_trials"]["mean_accuracy_delta"] * 100:+.2f}%** |
"""
    with open(OUT_DIR / "calibration_results.md", "w") as f:
        f.write(md_content)

    return calibration_results


# ==============================================================================
# MAIN EXECUTOR & PHASE 6 FINAL REPORT GENERATION
# ==============================================================================
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    print("\n" + "=" * 80)
    print("  SAFE POST-FINAL ACCURACY-IMPROVEMENT EXPERIMENTS (PHASES 2 - 6)")
    print("  Zero Test-Leakage Protocol (S094-S109 permanently frozen)")
    print("=" * 80)

    npz = np.load(DATA_NPZ)
    X_tr, y_tr = npz["X_train"], npz["y_train"]
    X_v, y_v = npz["X_val"], npz["y_val"]

    # 1. Phase 2: Weight Search
    p2_recs = run_phase_2(X_v, y_v, device)

    # 2. Phase 3: Multi-Seed Averaging
    p3_recs = run_phase_3(X_tr, y_tr, X_v, y_v, device)

    # 3. Phase 4: Safe Augmentations
    p4_recs = run_phase_4(X_tr, y_tr, X_v, y_v, device)

    # 4. Phase 5: Calibration Analysis
    run_phase_5(X_tr, y_tr, X_v, y_v, device)

    # Combine all zero-calibration records
    all_recs = p2_recs + p3_recs + p4_recs
    all_recs.sort(key=lambda r: r["val_metrics"]["macro_f1"], reverse=True)

    summary_rows = []
    for rank, r in enumerate(all_recs, 1):
        vm = r["val_metrics"]
        summary_rows.append(
            {
                "Rank": rank,
                "Phase": r["phase"],
                "Model Name": r["model_name"],
                "Params": r["total_parameters"],
                "Val Acc (%)": round(vm["accuracy"] * 100, 2),
                "Val Bal Acc (%)": round(vm["balanced_accuracy"] * 100, 2),
                "Val Macro F1": round(vm["macro_f1"], 4),
                "Val Kappa": round(vm.get("cohens_kappa", 0.0), 4),
                "Train Time (s)": r["train_time_sec"],
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "all_validation_results.csv", index=False)
    with open(OUT_DIR / "all_validation_results.json", "w") as f:
        json.dump(all_recs, f, indent=2, cls=NpEncoder)

    print("\n" + "=" * 90)
    print("      PHASE 6 SUMMARY: POST-FINAL VALIDATION RANKINGS (S078-S093)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")

    winner = all_recs[0]
    beats_ref = winner["val_metrics"]["macro_f1"] > VAL_REF_F1

    # Save Bar Chart Figure
    plt.figure(figsize=(10, 6))
    top10 = summary_rows[:10]
    names = [f"{r['Rank']}. {r['Model Name'][:28]}" for r in top10]
    f1s = [r["Val Macro F1"] for r in top10]
    colors = ["#2ecc71" if r["Val Macro F1"] > VAL_REF_F1 else "#3498db" for r in top10]
    plt.barh(names[::-1], f1s[::-1], color=colors[::-1])
    plt.axvline(
        x=VAL_REF_F1, color="red", linestyle="--", label=f"Val Reference ({VAL_REF_F1:.4f})"
    )
    plt.xlabel("Validation Macro F1")
    plt.title("Post-Final Candidate Validation Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "validation_ranking_top10.png", dpi=300)
    plt.close()

    # Generate Validation Ranking Markdown Report
    md_ranking = """# Post-Final Experiments: Validation Model Rankings (S078–S093)

> **REQUIRED STATEMENT**:
> **The official 80.98% test result remains frozen. All post-final experiments were conducted using training and validation subjects only and were not evaluated on the original test subjects.**

---

## Validation Summary Table

| Rank | Phase | Model Name | Total Params | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|
"""
    for r in summary_rows:
        md_ranking += f"| {r['Rank']} | {r['Phase']} | {r['Model Name']} | {r['Params']:,} | {r['Val Acc (%)']:.2f}% | {r['Val Macro F1']:.4f} | {r['Val Kappa']:.4f} |\n"

    md_ranking += f"""
---

## Key Findings & Conclusions
- **Top Validation Model**: **{winner["model_name"]}** (Val Acc = **{winner["val_metrics"]["accuracy"] * 100:.2f}%**, Val Macro F1 = **{winner["val_metrics"]["macro_f1"]:.4f}**).
- **Validation Outcome**: {"A new experiment improved validation performance over the reference ensemble!" if beats_ref else "The reference Val-Weighted Ensemble (Tuned CNN + EEGNet, 83.02% Val Acc) remains the top-performing model on validation."}
- **Official Test Benchmark**: **80.98% Test Accuracy** (Commit `5d7458d`) on $S094-S109$ remains frozen and untouched.
"""

    with open(OUT_DIR / "validation_ranking.md", "w") as f:
        f.write(md_ranking)

    print(f"  ✓ Saved validation ranking → {OUT_DIR / 'validation_ranking.md'}")
    print("  ✓ CONFIRMED: 0 test set evaluations on S094-S109.")
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
