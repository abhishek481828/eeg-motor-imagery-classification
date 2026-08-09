#!/usr/bin/env python3
"""Controlled CNN-LSTM Hyperparameter Tuning Experiment (Strict 2-Phase Protocol).

Phase 1 (Validation & Selection Phase):
  - Train all 7 configurations on training subjects (S001-S077).
  - Record validation metrics (S078-S093).
  - Select best checkpoint for each config using validation macro F1 ONLY.
  - DO NOT load or evaluate on the test set during this phase.

Phase 2 (Final Evaluation Phase):
  - Rank all 7 configurations by validation macro F1.
  - Select the SINGLE WINNING configuration (highest validation macro F1).
  - Freeze that single selected configuration.
  - Evaluate the selected configuration on test subjects (S094-S109) EXACTLY ONCE.
  - Compare its final test metrics against the frozen CNN baseline (72.81% test accuracy, 0.7856 val macro F1).

Configs:
    0  Original benchmark (reference)
    1  Hidden size: 128 → 64
    2  Dropout: 0.5 → 0.3
    3  Learning rate: 0.001 → 0.0003
    4  Gradient clipping: None → 1.0
    5  LSTM layers: 2 → 1
    6  ES patience: 10 → 20, max_epochs: 30 → 50
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.models.factory import create_model
from eeg_mi.training.seed import set_seed
from eeg_mi.training.trainer import Trainer
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("CNNLSTMTuning")

# ── Frozen paths ───────────────────────────────────────────────────────────────
DATA_NPZ      = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META     = ROOT / "data" / "processed" / "full_metadata.json"
OUT_DIR       = ROOT / "reports" / "experiments" / "cnn_lstm_tuning"
ORIG_CKPT     = ROOT / "models" / "checkpoints" / "full_cnn_lstm_2class_best.pt"  # READ-ONLY
CNN_BASELINE  = ROOT / "reports" / "experiments" / "full_dataset_best_model_test_report.json"

# ── Frozen reference thresholds (CNN Baseline) ────────────────────────────────
CNN_BASELINE_VAL_F1   = 0.7856   # benchmark val_macro_f1
CNN_BASELINE_TEST_ACC = 0.7281   # benchmark test accuracy

# ── Fixed parameters (identical across ALL configs) ───────────────────────────
FIXED = dict(
    dataset       = str(DATA_NPZ),
    split         = "S001-S077 train / S078-S093 val / S094-S109 test",
    input_shape   = (64, 481),
    num_channels  = 64,
    num_classes   = 2,
    labels        = "T1=class0 (Left Fist), T2=class1 (Right Fist)",
    optimizer     = "Adam",
    weight_decay  = 1e-4,
    lr_scheduler  = "ReduceLROnPlateau(mode=min, patience=5)",
    criterion     = "CrossEntropyLoss",
    batch_size    = 32,
    seed          = 42,
    gan_aug       = False,
    model_type    = "cnn_lstm",
)

# ── Config definitions (one change at a time) ─────────────────────────────────
CONFIGS: dict[str, dict] = {
    "0": dict(
        name        = "config_0_original",
        description = "Original benchmark config — reference. All defaults.",
        change      = "none",
        lstm_hidden = 128,
        lstm_layers = 2,
        dropout     = 0.5,
        lr          = 0.001,
        grad_clip   = None,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "1": dict(
        name        = "config_1_hidden64",
        description = "Hidden size reduced: 128 → 64.",
        change      = "lstm_hidden_size: 128 → 64",
        lstm_hidden = 64,
        lstm_layers = 2,
        dropout     = 0.5,
        lr          = 0.001,
        grad_clip   = None,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "2": dict(
        name        = "config_2_dropout03",
        description = "Dropout reduced: 0.5 → 0.3.",
        change      = "dropout: 0.5 → 0.3",
        lstm_hidden = 128,
        lstm_layers = 2,
        dropout     = 0.3,
        lr          = 0.001,
        grad_clip   = None,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "3": dict(
        name        = "config_3_lr0003",
        description = "Learning rate reduced: 0.001 → 0.0003.",
        change      = "lr: 0.001 → 0.0003",
        lstm_hidden = 128,
        lstm_layers = 2,
        dropout     = 0.5,
        lr          = 0.0003,
        grad_clip   = None,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "4": dict(
        name        = "config_4_gradclip1",
        description = "Gradient clipping added: clip_norm=1.0.",
        change      = "grad_clip: None → 1.0",
        lstm_hidden = 128,
        lstm_layers = 2,
        dropout     = 0.5,
        lr          = 0.001,
        grad_clip   = 1.0,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "5": dict(
        name        = "config_5_lstm1layer",
        description = "LSTM layers reduced: 2 → 1.",
        change      = "lstm_layers: 2 → 1",
        lstm_hidden = 128,
        lstm_layers = 1,
        dropout     = 0.5,
        lr          = 0.001,
        grad_clip   = None,
        es_patience = 10,
        max_epochs  = 30,
    ),
    "6": dict(
        name        = "config_6_patience20",
        description = "ES patience extended: 10 → 20, max_epochs: 30 → 50.",
        change      = "es_patience: 10 → 20, max_epochs: 30 → 50",
        lstm_hidden = 128,
        lstm_layers = 2,
        dropout     = 0.5,
        lr          = 0.001,
        grad_clip   = None,
        es_patience = 20,
        max_epochs  = 50,
    ),
}


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


CLASS_NAMES = ["Left Fist", "Right Fist"]


def _per_subject_breakdown(y_test, preds, meta):
    test_subs   = meta["subject_splits"]["test"]
    records     = meta.get("records_metadata", [])
    sub_counts  = {int(s): 0 for s in test_subs}
    for rec in records:
        sid = rec.get("subject_id") or rec.get("subject", 0)
        if isinstance(sid, str) and sid.startswith("S"):
            sid = int(sid[1:])
        if int(sid) in sub_counts:
            sub_counts[int(sid)] += 1
    breakdown = {}
    offset = 0
    for s in test_subs:
        s_int = int(s)
        s_str = f"S{s_int:03d}"
        n_ep  = sub_counts.get(s_int, 0)
        s_y   = y_test[offset : offset + n_ep]
        s_p   = preds[offset : offset + n_ep]
        offset += n_ep
        s_acc = float(np.mean(s_y == s_p)) if len(s_y) > 0 else 0.0
        breakdown[s_str] = {
            "num_epochs": int(len(s_y)),
            "accuracy": round(s_acc, 6),
        }
    return breakdown


# ── PHASE 1: Training & Validation Only (No Test Evaluation) ─────────────────

def train_and_validate_config(
    cfg_key: str,
    X_train, y_train, X_val, y_val,
    device: torch.device,
) -> dict[str, Any]:
    cfg = CONFIGS[cfg_key]
    config_dir = OUT_DIR / cfg["name"]
    config_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = config_dir / "checkpoint.pt"

    print(f"\n{'='*72}")
    print(f"  [PHASE 1: TRAIN & VAL] CONFIG {cfg_key}: {cfg['name']}")
    print(f"  Change: {cfg['change']}")
    print(f"  Description: {cfg['description']}")
    print(f"{'='*72}")

    # 1. Seed
    set_seed(FIXED["seed"])

    # 2. Model
    seq_len = int(X_train.shape[2])
    model = create_model(
        "cnn_lstm",
        num_channels    = FIXED["num_channels"],
        num_classes     = FIXED["num_classes"],
        sequence_length = seq_len,
        lstm_hidden_size = cfg["lstm_hidden"],
        lstm_layers      = cfg["lstm_layers"],
        dropout          = cfg["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")

    # 3. DataLoaders (Train and Val ONLY)
    bs = FIXED["batch_size"]
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),   batch_size=bs, shuffle=False, num_workers=0)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=FIXED["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=ckpt_path,
        scheduler=scheduler,
        patience=cfg["es_patience"],
        grad_clip=cfg["grad_clip"],
        config_dict={
            "config_key": cfg_key,
            "name": cfg["name"],
            "change": cfg["change"],
            **{k: cfg[k] for k in ("lstm_hidden", "lstm_layers", "dropout", "lr", "grad_clip", "es_patience", "max_epochs")},
        },
    )

    # 6. Fit (Train and Validate ONLY)
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=cfg["max_epochs"])
    train_time = round(time.time() - t0, 2)

    best_val_f1  = max(history["val_macro_f1"])
    best_val_acc = max(history["val_acc"])
    best_val_loss = min(history["val_loss"])
    print(f"  ✓ Phase 1 Complete in {train_time}s | Best Val F1={best_val_f1:.4f} | Best Val Acc={best_val_acc:.4f}")

    # Load checkpoint to get best epoch
    saved = torch.load(ckpt_path, map_location="cpu")
    best_epoch = int(saved.get("epoch", -1))

    val_record = {
        "config_key": cfg_key,
        "name": cfg["name"],
        "description": cfg["description"],
        "change": cfg["change"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixed_parameters": FIXED,
        "tuned_parameters": {
            "lstm_hidden_size":  cfg["lstm_hidden"],
            "lstm_layers":       cfg["lstm_layers"],
            "dropout":           cfg["dropout"],
            "lr":                cfg["lr"],
            "grad_clip":         cfg["grad_clip"],
            "es_patience":       cfg["es_patience"],
            "max_epochs":        cfg["max_epochs"],
        },
        "total_parameters":      total_params,
        "best_checkpoint_epoch": best_epoch,
        "best_val_macro_f1":     round(best_val_f1, 6),
        "best_val_acc":          round(best_val_acc, 6),
        "best_val_loss":         round(best_val_loss, 6),
        "train_time_sec":        train_time,
        "test_evaluated":        False,
        "model_selection":       "validation_macro_f1_only",
    }

    with open(config_dir / "val_results.json", "w") as f:
        json.dump(val_record, f, indent=2, cls=NpEncoder)

    return val_record


# ── PHASE 2: Single Winner Selection & Final Test Evaluation ───────────────────

def evaluate_winning_config_on_test(
    winning_cfg_key: str,
    X_train, X_test, y_test,
    meta: dict,
    device: torch.device,
) -> dict[str, Any]:
    cfg = CONFIGS[winning_cfg_key]
    config_dir = OUT_DIR / cfg["name"]
    ckpt_path  = config_dir / "checkpoint.pt"

    print("\n" + "=" * 72)
    print(f"  [PHASE 2: FINAL TEST EVALUATION] WINNING CONFIG {winning_cfg_key}: {cfg['name']}")
    print(f"  Selected strictly by Validation Macro F1")
    print("=" * 72)

    seq_len = int(X_train.shape[2])
    model = create_model(
        "cnn_lstm",
        num_channels    = FIXED["num_channels"],
        num_classes     = FIXED["num_classes"],
        sequence_length = seq_len,
        lstm_hidden_size = cfg["lstm_hidden"],
        lstm_layers      = cfg["lstm_layers"],
        dropout          = cfg["dropout"],
    )

    # Load frozen best checkpoint
    saved = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    model.eval()

    bs = FIXED["batch_size"]
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=bs, shuffle=False, num_workers=0)

    t_inf = time.time()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    infer_time = round(time.time() - t_inf, 4)
    preds = np.array(preds)

    test_metrics = compute_metrics(y_test, preds, class_names=CLASS_NAMES)
    per_subject  = _per_subject_breakdown(y_test, preds, meta)
    sub_accs     = [v["accuracy"] for v in per_subject.values() if v["num_epochs"] > 0]

    test_acc = test_metrics["accuracy"]
    val_f1   = saved.get("metrics", {}).get("val_macro_f1", 0.0)

    beats_val  = val_f1 > CNN_BASELINE_VAL_F1
    beats_test = test_acc > CNN_BASELINE_TEST_ACC
    if beats_val and beats_test:
        verdict = "IMPROVEMENT ✓ — beats CNN baseline on both val AND test"
    elif beats_val:
        verdict = f"PARTIAL — beats val F1 ({val_f1:.4f} > {CNN_BASELINE_VAL_F1}) but not test"
    elif beats_test:
        verdict = f"PARTIAL — beats test acc ({test_acc*100:.2f}% > {CNN_BASELINE_TEST_ACC*100:.2f}%) but not val"
    else:
        verdict = "NO IMPROVEMENT — tuned CNN-LSTM remains below frozen CNN baseline"

    cm = np.array(test_metrics["confusion_matrix"])
    np.save(config_dir / "confusion_matrix.npy", cm)
    with open(config_dir / "per_subject.json", "w") as f:
        json.dump(per_subject, f, indent=2, cls=NpEncoder)

    final_report = {
        "winning_config_key": winning_cfg_key,
        "name": cfg["name"],
        "description": cfg["description"],
        "change": cfg["change"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Highest Validation Macro F1 among all 7 configs",
        "validation_macro_f1": round(val_f1, 6),
        "test_metrics": test_metrics,
        "per_subject_accuracy_mean": round(float(np.mean(sub_accs)), 6),
        "per_subject_accuracy_std":  round(float(np.std(sub_accs)),  6),
        "cnn_baseline_val_f1_threshold":   CNN_BASELINE_VAL_F1,
        "cnn_baseline_test_acc_threshold": CNN_BASELINE_TEST_ACC,
        "verdict": verdict,
        "test_evaluated_once": True,
        "gan_augmentation": False,
    }

    with open(OUT_DIR / "winning_config_test_report.json", "w") as f:
        json.dump(final_report, f, indent=2, cls=NpEncoder)

    print(f"\n  FINAL TEST RESULTS FOR WINNER ({cfg['name']}):")
    print(f"  Test Accuracy   : {test_acc*100:.2f}%")
    print(f"  Test Macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Kappa      : {test_metrics['cohens_kappa']:.4f}")
    print(f"  Verdict         : {verdict}")

    return final_report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="CNN-LSTM Controlled Tuning Experiment")
    parser.add_argument(
        "--config", "-c", default="all",
        help="Config key to run (0-6) or 'all' to run all sequentially"
    )
    args = parser.parse_args()

    assert ORIG_CKPT.exists(), f"Original CNN-LSTM checkpoint missing: {ORIG_CKPT}"

    print("\n" + "=" * 78)
    print("  CNN-LSTM CONTROLLED TUNING EXPERIMENT (2-PHASE SELECTION PROTOCOL)")
    print(f"  GAN augmentation : DISABLED")
    print(f"  Frozen baselines : CNN val F1={CNN_BASELINE_VAL_F1}  test acc={CNN_BASELINE_TEST_ACC*100:.2f}%")
    print(f"  Original LSTM ckpt (untouched): {ORIG_CKPT}")
    print("=" * 78)

    # Load dataset
    print("\n  Loading full_dataset.npz...")
    npz = np.load(DATA_NPZ)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val,   y_val   = npz["X_val"],   npz["y_val"]
    X_test,  y_test  = npz["X_test"],  npz["y_test"]
    with open(DATA_META) as f:
        meta = json.load(f)

    device = get_device("auto")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.config == "all":
        keys = sorted(CONFIGS.keys())
    else:
        keys = [args.config]

    # ── PHASE 1: Train & Validate All Configs ─────────────────────────────────
    phase1_results = []
    for key in keys:
        res = train_and_validate_config(key, X_train, y_train, X_val, y_val, device)
        phase1_results.append(res)

    # Build Validation Summary Table
    rows = []
    for r in phase1_results:
        rows.append({
            "Config Key": r["config_key"],
            "Config Name": r["name"],
            "Change": r["change"],
            "Params": r["total_parameters"],
            "Best Epoch": r["best_checkpoint_epoch"],
            "Val Macro F1": round(r["best_val_macro_f1"], 4),
            "Val Accuracy (%)": round(r["best_val_acc"] * 100, 2),
            "Train Time (s)": r["train_time_sec"],
        })
    df_val = pd.DataFrame(rows)
    df_val.to_csv(OUT_DIR / "val_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("  PHASE 1 SUMMARY: VALIDATION RANKING (TEST SET UNTOUCHED)")
    print("=" * 78)
    print(df_val.to_string(index=False))
    print("=" * 78)

    # ── PHASE 2: Select Best Winner & Test Evaluate Exactly Once ───────────────
    if args.config == "all":
        winner = max(phase1_results, key=lambda x: x["best_val_macro_f1"])
        winning_key = winner["config_key"]
        print(f"\n  🏆 SELECTED WINNER BY VALIDATION MACRO F1: Config {winning_key} ({winner['name']})")
        print(f"     Validation Macro F1: {winner['best_val_macro_f1']:.4f}")

        final_report = evaluate_winning_config_on_test(winning_key, X_train, X_test, y_test, meta, device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
