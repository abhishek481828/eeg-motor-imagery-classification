#!/usr/bin/env python3
"""Multi-Seed CNN Baseline Reproducibility Study.

Trains 5 independent CNN copies (one per seed) using the exact frozen
architecture and hyperparameters from the benchmark.  The existing baseline
checkpoint is NEVER modified.

Seeds: 42, 123, 2024, 777, 999

Usage:
    python scripts/reproduce_cnn_seeds.py

Outputs (all under reports/experiments/seed_study/):
    seed_042/
        checkpoint_seed042.pt
        test_metrics_seed042.json
        confusion_matrix_seed042.npy
        per_subject_seed042.json
    reproducibility_report.json
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.factory import create_model
from eeg_mi.training.callbacks import EarlyStopping, ModelCheckpoint
from eeg_mi.training.seed import set_seed
from eeg_mi.training.trainer import Trainer
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("SeedStudy")

# ── Frozen paths — READ ONLY ──────────────────────────────────────────────────
DATA_NPZ    = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META   = ROOT / "data" / "processed" / "full_metadata.json"
FROZEN_CKPT = ROOT / "models" / "checkpoints" / "full_cnn_baseline_best.pt"  # READ-ONLY
EXISTING_REPORT = ROOT / "reports" / "experiments" / "full_dataset_best_model_test_report.json"

# ── Output dir ────────────────────────────────────────────────────────────────
SEED_STUDY_DIR = ROOT / "reports" / "experiments" / "seed_study"

# ── Frozen hyperparameters — EXACT COPY from benchmark_full_dataset.py ────────
# Source: scripts/benchmark_full_dataset.py lines 267–290
# ANY change to these values makes this a NEW configuration, not a reproduction.
SEEDS          = [42, 123, 2024, 777, 999]
MAX_EPOCHS     = 30      # benchmark line 290: epochs=30
BATCH_SIZE     = 32      # benchmark line 267: batch_sz = 32
LR             = 0.001   # benchmark line 275: lr=0.001
WEIGHT_DECAY   = 1e-4    # benchmark line 275: weight_decay=1e-4
LR_PATIENCE    = 5       # benchmark line 277: ReduceLROnPlateau patience=5
ES_PATIENCE    = 10      # benchmark line 287: Trainer patience=10  ← was incorrectly 15, now fixed
CLASS_NAMES    = ["Left Fist", "Right Fist"]


# ── Helpers ───────────────────────────────────────────────────────────────────

class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def _get_pred_dist(preds: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[int(k)]: int(v)
            for k, v in zip(*np.unique(preds, return_counts=True))}


def _per_subject_breakdown(
    y_test: np.ndarray,
    preds: np.ndarray,
    meta: dict,
) -> dict[str, dict]:
    """Compute per-subject accuracy using actual epoch counts from metadata."""
    test_subs  = meta["subject_splits"]["test"]
    records    = meta.get("records_metadata", [])
    sub_counts: dict[int, int] = {int(s): 0 for s in test_subs}
    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject", 0)
        if isinstance(sid, str) and sid.startswith("S"):
            sid = int(sid[1:])
        if int(sid) in sub_counts:
            sub_counts[int(sid)] += 1

    breakdown: dict[str, dict] = {}
    offset = 0
    for s in test_subs:
        s_int = int(s)
        s_str = f"S{s_int:03d}"
        n_ep  = sub_counts.get(s_int, 0)
        s_y   = y_test[offset : offset + n_ep]
        s_p   = preds[offset : offset + n_ep]
        offset += n_ep
        s_acc = float(np.mean(s_y == s_p)) if len(s_y) > 0 else 0.0
        if len(s_y) > 0:
            uniq, cnts = np.unique(s_y, return_counts=True)
            cls_dist = {str(int(k)): int(v) for k, v in zip(uniq, cnts)}
        else:
            cls_dist = {}
        breakdown[s_str] = {
            "num_epochs": int(len(s_y)),
            "accuracy":   round(s_acc, 6),
            "class_distribution": cls_dist,
        }
    return breakdown


# ── Per-seed training ─────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    meta: dict,
    device: torch.device,
) -> dict[str, Any]:
    """Train one fresh CNN with given seed, evaluate on val for selection, test once."""

    seed_str = f"{seed:04d}" if seed < 10000 else str(seed)
    out_dir  = SEED_STUDY_DIR / f"seed_{seed_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / f"checkpoint_seed{seed_str}.pt"

    print(f"\n{'─'*70}")
    print(f"  Seed {seed}  |  output → {out_dir.relative_to(ROOT)}")
    print(f"{'─'*70}")

    # 1. Deterministic seeding
    set_seed(seed)

    # 2. Data loaders
    seq_len      = int(X_train.shape[2])
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(EEGDataset(X_test,  y_test),  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # 3. Fresh model (same frozen architecture — zero weights from existing checkpoint)
    model = create_model("cnn", num_channels=64, num_classes=2, sequence_length=seq_len)

    # 4. Optimiser, loss, LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE
    )

    # 5. Trainer (model selection on val_macro_f1 only — never touches test)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=ckpt_path,
        scheduler=scheduler,
        patience=ES_PATIENCE,
        config_dict={"seed": seed, "max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE},
    )

    t_train = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
    train_time = round(time.time() - t_train, 2)

    best_val_f1  = max(history["val_macro_f1"])
    best_val_acc = max(history["val_acc"])
    print(f"  Training done in {train_time}s | Best Val F1={best_val_f1:.4f} | Best Val Acc={best_val_acc:.4f}")

    # 6. Load best checkpoint (selected by val_macro_f1) and evaluate TEST ONCE
    saved_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(saved_ckpt["state_dict"])
    model.to(device)
    model.eval()

    t_infer = time.time()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    infer_time = round(time.time() - t_infer, 4)
    preds = np.array(preds)

    # 7. Compute all metrics from raw predictions
    test_metrics    = compute_metrics(y_test, preds, class_names=CLASS_NAMES)
    pred_dist       = _get_pred_dist(preds)
    per_subject     = _per_subject_breakdown(y_test, preds, meta)
    sub_accs        = [v["accuracy"] for v in per_subject.values() if v["num_epochs"] > 0]

    # 8. Save confusion matrix
    cm = np.array(test_metrics["confusion_matrix"])
    np.save(out_dir / f"confusion_matrix_seed{seed_str}.npy", cm)

    # 9. Save per-subject breakdown
    with open(out_dir / f"per_subject_seed{seed_str}.json", "w") as f:
        json.dump(per_subject, f, indent=2, cls=NpEncoder)

    # 10. Save full test metrics
    seed_result = {
        "seed": seed,
        "best_epoch": int(saved_ckpt.get("epoch", -1)),
        "best_val_macro_f1": round(best_val_f1, 6),
        "best_val_acc":      round(best_val_acc, 6),
        "train_time_sec":    train_time,
        "infer_time_sec":    infer_time,
        "test_metrics":      test_metrics,
        "prediction_distribution": pred_dist,
        "per_subject_accuracy_mean": round(float(np.mean(sub_accs)), 6),
        "per_subject_accuracy_std":  round(float(np.std(sub_accs)),  6),
        "model_selection_criterion": "validation_macro_f1_only",
        "test_evaluated_once": True,
        "frozen_architecture": True,
        "gan_augmentation": False,
    }
    with open(out_dir / f"test_metrics_seed{seed_str}.json", "w") as f:
        json.dump(seed_result, f, indent=2, cls=NpEncoder)

    print(f"  Test Accuracy={test_metrics['accuracy']*100:.2f}%  "
          f"Macro F1={test_metrics['macro_f1']:.4f}  "
          f"Kappa={test_metrics['cohens_kappa']:.4f}")

    return seed_result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_seed_study() -> int:
    # Safety: confirm existing report is NOT overwritten
    repro_out = SEED_STUDY_DIR / "reproducibility_report.json"
    if EXISTING_REPORT.exists() and repro_out.exists():
        assert EXISTING_REPORT.resolve() != repro_out.resolve(), \
            "Output path would overwrite existing baseline report!"

    print("\n" + "=" * 78)
    print("  MULTI-SEED CNN BASELINE REPRODUCIBILITY STUDY")
    print(f"  Seeds      : {SEEDS}")
    print(f"  Frozen arch: CNN Baseline  (num_channels=64, num_classes=2)")
    print(f"  Max epochs : {MAX_EPOCHS}  |  batch_size={BATCH_SIZE}  |  ES patience={ES_PATIENCE} (exact match: benchmark line 287)")
    print(f"  GAN aug    : DISABLED")
    print(f"  Frozen ckpt (reference only): {FROZEN_CKPT}")
    print(f"  Existing report (preserved):  {EXISTING_REPORT}")
    print("=" * 78)

    # Load data once (read-only)
    print("\n  Loading full_dataset.npz ...")
    npz = np.load(DATA_NPZ)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val,   y_val   = npz["X_val"],   npz["y_val"]
    X_test,  y_test  = npz["X_test"],  npz["y_test"]
    with open(DATA_META) as f:
        meta = json.load(f)

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Epoch shape: (64, {X_train.shape[2]})")

    device = get_device("auto")
    SEED_STUDY_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_results: dict[str, Any] = {}

    for seed in SEEDS:
        result = run_one_seed(
            seed, X_train, y_train, X_val, y_val, X_test, y_test, meta, device
        )
        all_results[str(seed)] = result

    total_time = round(time.time() - t0, 1)

    # Aggregate statistics
    accs   = [all_results[str(s)]["test_metrics"]["accuracy"]        for s in SEEDS]
    f1s    = [all_results[str(s)]["test_metrics"]["macro_f1"]        for s in SEEDS]
    kappas = [all_results[str(s)]["test_metrics"]["cohens_kappa"]     for s in SEEDS]
    bal    = [all_results[str(s)]["test_metrics"]["balanced_accuracy"] for s in SEEDS]

    agg = {
        "mean_test_accuracy":          round(float(np.mean(accs)),    6),
        "std_test_accuracy":           round(float(np.std(accs)),     6),
        "mean_macro_f1":               round(float(np.mean(f1s)),     6),
        "std_macro_f1":                round(float(np.std(f1s)),      6),
        "mean_cohens_kappa":           round(float(np.mean(kappas)),  6),
        "std_cohens_kappa":            round(float(np.std(kappas)),   6),
        "mean_balanced_accuracy":      round(float(np.mean(bal)),     6),
        "std_balanced_accuracy":       round(float(np.std(bal)),      6),
        "min_test_accuracy":           round(float(np.min(accs)),     6),
        "max_test_accuracy":           round(float(np.max(accs)),     6),
    }

    repro_report = {
        "study_timestamp":  datetime.now(timezone.utc).isoformat(),
        "total_time_sec":   total_time,
        "seeds":            SEEDS,
        "frozen_checkpoint_reference": str(FROZEN_CKPT.resolve()),
        "existing_baseline_report":    str(EXISTING_REPORT.resolve()),
        "existing_report_preserved":   True,
        "model_selection_criterion":   "validation_macro_f1_only",
        "test_evaluated_once_per_seed": True,
        "gan_augmentation":            False,
        "config_matches_original": True,
        "config_source": "scripts/benchmark_full_dataset.py lines 267-290",
        "config_correction_note": "ES_PATIENCE was incorrectly set to 15 in first draft; corrected to 10 to match benchmark line 287 before any seed ran to completion",
        "frozen_hyperparameters": {
            "architecture":    "CNN Baseline",
            "num_channels":    64,
            "num_classes":     2,
            "sequence_length": int(X_train.shape[2]),
            "optimizer":       "Adam",
            "lr":              LR,
            "weight_decay":    WEIGHT_DECAY,
            "max_epochs":      MAX_EPOCHS,
            "batch_size":      BATCH_SIZE,
            "early_stopping_patience": ES_PATIENCE,
        },
        "aggregate": agg,
        "per_seed":  all_results,
    }

    repro_path = SEED_STUDY_DIR / "reproducibility_report.json"
    with open(repro_path, "w") as f:
        json.dump(repro_report, f, indent=2, cls=NpEncoder)

    # Final summary table
    print("\n" + "=" * 78)
    print("  REPRODUCIBILITY STUDY — FINAL SUMMARY")
    print("=" * 78)
    print(f"  {'Seed':>6}  {'Test Acc':>9}  {'Macro F1':>9}  {'Kappa':>8}  {'Bal Acc':>9}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*9}")
    for s in SEEDS:
        r = all_results[str(s)]["test_metrics"]
        print(f"  {s:>6}  {r['accuracy']*100:8.2f}%  {r['macro_f1']:9.4f}  {r['cohens_kappa']:8.4f}  {r['balanced_accuracy']*100:8.2f}%")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*9}")
    print(f"  {'MEAN':>6}  {agg['mean_test_accuracy']*100:8.2f}%  {agg['mean_macro_f1']:9.4f}  {agg['mean_cohens_kappa']:8.4f}  {agg['mean_balanced_accuracy']*100:8.2f}%")
    print(f"  {'STD':>6}  {agg['std_test_accuracy']*100:8.2f}%  {agg['std_macro_f1']:9.4f}  {agg['std_cohens_kappa']:8.4f}  {agg['std_balanced_accuracy']*100:8.2f}%")
    print("=" * 78)
    print(f"\n  Total study time : {total_time}s")
    print(f"  Report saved     → {repro_path.resolve()}")
    print(f"  Existing report preserved: {EXISTING_REPORT}")
    print(f"  Frozen checkpoint intact:  {FROZEN_CKPT}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_seed_study())
