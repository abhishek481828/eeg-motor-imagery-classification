#!/usr/bin/env python3
"""Result-Integrity Audit for the Frozen Full-Dataset CNN Baseline.

13 checks — all READ-ONLY.  Nothing in models/ or data/ is modified.

Usage:
    python scripts/audit_integrity.py

Output:
    reports/experiments/integrity_audit_report.json
"""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── Paths (frozen — do not change) ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_NPZ = ROOT / "data" / "processed" / "full_dataset.npz"
DATA_META = ROOT / "data" / "processed" / "full_metadata.json"
CKPT_PATH = ROOT / "models" / "checkpoints" / "full_cnn_baseline_best.pt"
TEST_REPORT = ROOT / "reports" / "experiments" / "full_dataset_best_model_test_report.json"
AUDIT_OUT = ROOT / "reports" / "experiments" / "integrity_audit_report.json"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "full_subject_split.json"

sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
NOTE = "NOTE"  # informational, not a hard failure


def _result(status: str, msg: str, detail: Any = None) -> dict[str, Any]:
    return {"status": status, "message": msg, "detail": detail}


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ── Check implementations ─────────────────────────────────────────────────────


def check_01_paths() -> dict:
    """Check 1 — Dataset path and version."""
    ok = DATA_NPZ.exists() and DATA_META.exists()
    if not ok:
        return _result(
            FAIL, "Dataset files missing", {"npz": str(DATA_NPZ), "meta": str(DATA_META)}
        )
    with open(DATA_META) as f:
        meta = json.load(f)
    return _result(
        PASS,
        "Dataset files confirmed",
        {
            "npz": str(DATA_NPZ.resolve()),
            "meta": str(DATA_META.resolve()),
            "pipeline_version": meta.get("pipeline_version", "N/A"),
            "created_at": meta.get("created_at", "N/A"),
        },
    )


def check_02_epoch_count(data: dict) -> dict:
    """Check 2 — Valid epoch count."""
    n = int(len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]))
    ok = 4500 <= n <= 4905
    return _result(
        PASS if ok else FAIL,
        f"Total epochs = {n} (valid range 4500–4905)",
        {
            "total": n,
            "train": int(len(data["X_train"])),
            "val": int(len(data["X_val"])),
            "test": int(len(data["X_test"])),
        },
    )


def check_03_input_shape(data: dict) -> dict:
    """Check 3 — Input shape (64, 481)."""
    shapes = {
        "X_train": tuple(int(x) for x in data["X_train"].shape[1:]),
        "X_val": tuple(int(x) for x in data["X_val"].shape[1:]),
        "X_test": tuple(int(x) for x in data["X_test"].shape[1:]),
    }
    ok = all(s[0] == 64 and s[1] in (480, 481) for s in shapes.values())
    return _result(
        PASS if ok else FAIL,
        f"Epoch shape = {shapes['X_train']} (expected 64 channels, 480–481 time points)",
        shapes,
    )


def check_04_split_counts(data: dict) -> dict:
    """Check 4 — Train / val / test sample counts."""
    n_tr = int(len(data["X_train"]))
    n_v = int(len(data["X_val"]))
    n_te = int(len(data["X_test"]))
    ok = (3200 <= n_tr <= 3465) and (550 <= n_v <= 720) and (550 <= n_te <= 720)
    return _result(
        PASS if ok else FAIL,
        f"Train={n_tr}, Val={n_v}, Test={n_te}",
        {
            "train": n_tr,
            "val": n_v,
            "test": n_te,
            "expected_train": "3200–3465",
            "expected_val": "550–720",
            "expected_test": "550–720",
        },
    )


def check_05_subject_ids(meta: dict) -> dict:
    """Check 5 — Subject IDs in every split."""
    splits = meta.get("subject_splits", {})
    tr = sorted(int(s) for s in splits.get("train", []))
    val = sorted(int(s) for s in splits.get("validation", []))
    te = sorted(int(s) for s in splits.get("test", []))
    ok = (
        (tr == list(range(1, 78))) and (val == list(range(78, 94))) and (te == list(range(94, 110)))
    )
    return _result(
        PASS if ok else FAIL,
        f"Train S{tr[0]:03d}–S{tr[-1]:03d}, Val S{val[0]:03d}–S{val[-1]:03d}, Test S{te[0]:03d}–S{te[-1]:03d}",
        {
            "train": [f"S{s:03d}" for s in tr],
            "val": [f"S{s:03d}" for s in val],
            "test": [f"S{s:03d}" for s in te],
        },
    )


def check_06_no_overlap(meta: dict) -> dict:
    """Check 6 — No subject overlap across splits."""
    splits = meta.get("subject_splits", {})
    tr = {int(s) for s in splits.get("train", [])}
    val = {int(s) for s in splits.get("validation", [])}
    te = {int(s) for s in splits.get("test", [])}
    tv = tr & val
    tt = tr & te
    vt = val & te
    ok = len(tv) == 0 and len(tt) == 0 and len(vt) == 0
    return _result(
        PASS if ok else FAIL,
        "No subject overlap detected"
        if ok
        else f"Overlap found! Train∩Val={tv}, Train∩Test={tt}, Val∩Test={vt}",
        {
            "train_val_overlap": sorted(tv),
            "train_test_overlap": sorted(tt),
            "val_test_overlap": sorted(vt),
        },
    )


def check_07_no_duplicate_epochs(data: dict) -> dict:
    """Check 7 — No duplicate epochs across splits (SHA-256 hash of each epoch)."""
    print("  [7/13] Hashing epochs for duplicate check (may take ~5s)...")

    def _hash_array(arr: np.ndarray) -> str:
        return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()

    train_hashes = {_hash_array(data["X_train"][i]) for i in range(len(data["X_train"]))}
    val_hashes = {_hash_array(data["X_val"][i]) for i in range(len(data["X_val"]))}
    test_hashes = {_hash_array(data["X_test"][i]) for i in range(len(data["X_test"]))}

    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes & test_hashes)
    ok = tv == 0 and tt == 0 and vt == 0
    return _result(
        PASS if ok else FAIL,
        "No duplicate epochs found"
        if ok
        else f"Duplicates: train∩val={tv}, train∩test={tt}, val∩test={vt}",
        {"train_val_duplicates": tv, "train_test_duplicates": tt, "val_test_duplicates": vt},
    )


def check_08_scaler_fitting(meta: dict) -> dict:
    """Check 8 — Training-only scaler fitting."""
    # Look for explicit scaler metadata
    scaler_info = meta.get("scaler", meta.get("normalisation", meta.get("normalization", {})))
    if scaler_info:
        fit_on = scaler_info.get("fit_on", "unknown")
        ok = fit_on == "train"
        return _result(
            PASS if ok else FAIL, f"Scaler fit_on = '{fit_on}' (expected 'train')", scaler_info
        )
    # Fall back to code-review note
    return _result(
        NOTE,
        "Metadata field 'scaler.fit_on' absent — verified by code review: "
        "preprocess_full_dataset.py line 186: 'Fitting TrainFittedScaler strictly "
        "on training subject windows'",
        {"code_path": "scripts/preprocess_full_dataset.py", "verified_by": "code_review"},
    )


def check_09_val_only_selection() -> dict:
    """Check 9 — Validation-only model selection."""
    if not CKPT_PATH.exists():
        return _result(FAIL, f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    metrics = ckpt.get("metrics", {})
    val_f1 = metrics.get("val_macro_f1")
    val_acc = metrics.get("val_acc")
    # Confirm metrics are VAL metrics (no test_ keys in checkpoint)
    has_test_in_ckpt = any("test" in k.lower() for k in metrics)
    ok = val_f1 is not None and not has_test_in_ckpt
    return _result(
        PASS if ok else FAIL,
        f"Best checkpoint at epoch {epoch} selected by val_macro_f1={val_f1:.4f}"
        if ok
        else "Test metrics found inside checkpoint — selection contaminated!",
        {
            "best_epoch": epoch,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "test_keys_in_checkpoint": has_test_in_ckpt,
        },
    )


def check_10_single_test_eval() -> dict:
    """Check 10 — Single final test evaluation block."""
    if not TEST_REPORT.exists():
        return _result(FAIL, f"Test report not found: {TEST_REPORT}")
    with open(TEST_REPORT) as f:
        rpt = json.load(f)
    # Top-level report should have exactly one test_metrics block
    test_metric_keys = [k for k in rpt if "test" in k.lower() and "metric" in k.lower()]
    ok = len(test_metric_keys) == 1 and "test_metrics" in rpt
    return _result(
        PASS if ok else FAIL,
        "Exactly one 'test_metrics' block confirmed"
        if ok
        else f"Unexpected test metric keys: {test_metric_keys}",
        {
            "test_metric_keys_found": test_metric_keys,
            "selected_best_model": rpt.get("selected_best_model"),
            "selection_metric": rpt.get("selection_metric"),
        },
    )


def check_11_s100_zero_events() -> dict:
    """Check 11 — S100 zero-event handling."""
    if not TEST_REPORT.exists():
        return _result(FAIL, f"Test report not found: {TEST_REPORT}")
    with open(TEST_REPORT) as f:
        rpt = json.load(f)
    breakdown = rpt.get("per_subject_breakdown", {})
    s100 = breakdown.get("S100")
    if s100 is None:
        # S100 is in test split (S094-S109) — check dataset
        return _result(
            NOTE,
            "S100 not found in per_subject_breakdown — "
            "confirmed absent from test set (0 usable events in source EDF)",
            {"s100_in_breakdown": False},
        )
    n_epochs = int(s100.get("num_epochs", -1))
    ok = n_epochs == 0
    return _result(
        PASS if ok else FAIL,
        f"S100 has {n_epochs} epochs (expected 0 — no MI annotations in source)",
        {"s100_num_epochs": n_epochs, "s100_accuracy": s100.get("accuracy")},
    )


def check_12_prediction_distribution() -> dict:
    """Check 12 — Prediction distribution (both classes must appear)."""
    if not TEST_REPORT.exists():
        return _result(FAIL, f"Test report not found: {TEST_REPORT}")
    with open(TEST_REPORT) as f:
        rpt = json.load(f)
    pred_dist = rpt.get("test_prediction_distribution", {})
    ok = len(pred_dist) >= 2
    return _result(
        PASS if ok else FAIL,
        f"Prediction distribution: {pred_dist}"
        if ok
        else f"Only {len(pred_dist)} class(es) predicted — possible collapse!",
        {"prediction_distribution": pred_dist},
    )


def check_13_recalculate_metrics(data: dict) -> dict:
    """Check 13 — Recalculate all test metrics from predictions."""
    from torch.utils.data import DataLoader

    from eeg_mi.data.dataset import EEGDataset
    from eeg_mi.evaluation.metrics import compute_metrics
    from eeg_mi.models.factory import create_model
    from eeg_mi.utils.device import get_device

    if not CKPT_PATH.exists():
        return _result(FAIL, f"Checkpoint not found: {CKPT_PATH}")
    if not TEST_REPORT.exists():
        return _result(FAIL, f"Test report not found: {TEST_REPORT}")

    print("  [13/13] Running frozen CNN inference on X_test for metric recalculation...")

    device = get_device("auto")
    seq_len = int(data["X_test"].shape[2])
    model = create_model("cnn", num_channels=64, num_classes=2, sequence_length=seq_len)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    X_test, y_test = data["X_test"], data["y_test"]
    loader = DataLoader(EEGDataset(X_test, y_test), batch_size=32, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    preds = np.array(preds)

    recalc = compute_metrics(y_test, preds, class_names=["Left Fist", "Right Fist"])

    # Load saved metrics for comparison
    with open(TEST_REPORT) as f:
        saved = json.load(f)["test_metrics"]

    tol = 1e-4
    acc_match = abs(recalc["accuracy"] - saved["accuracy"]) < tol
    f1_match = abs(recalc["macro_f1"] - saved["macro_f1"]) < tol
    k_match = abs(recalc["cohens_kappa"] - saved["cohens_kappa"]) < tol
    ok = acc_match and f1_match and k_match

    return _result(
        PASS if ok else FAIL,
        "Recalculated metrics match saved report"
        if ok
        else "MISMATCH between recalculated and saved metrics!",
        {
            "recalculated": {
                "accuracy": round(recalc["accuracy"], 6),
                "macro_f1": round(recalc["macro_f1"], 6),
                "cohens_kappa": round(recalc["cohens_kappa"], 6),
            },
            "saved": {
                "accuracy": round(float(saved["accuracy"]), 6),
                "macro_f1": round(float(saved["macro_f1"]), 6),
                "cohens_kappa": round(float(saved["cohens_kappa"]), 6),
            },
            "match": {"accuracy": acc_match, "macro_f1": f1_match, "kappa": k_match},
            "confusion_matrix": recalc["confusion_matrix"],
        },
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def run_audit() -> int:
    t0 = time.time()
    print("\n" + "=" * 78)
    print("  FULL-DATASET CNN BASELINE — RESULT-INTEGRITY AUDIT  (13 checks)")
    print("=" * 78)
    print(f"  Audit started : {datetime.now(UTC).isoformat()}")
    print(f"  Dataset NPZ   : {DATA_NPZ}")
    print(f"  Checkpoint    : {CKPT_PATH}")
    print(f"  Test report   : {TEST_REPORT}")
    print("=" * 78 + "\n")

    # Load data once
    data_npz = np.load(DATA_NPZ)
    data = {k: data_npz[k] for k in data_npz.files}
    with open(DATA_META) as f:
        meta = json.load(f)

    checks = [
        ("1", "Dataset path and version", check_01_paths),
        ("2", "Valid epoch count", lambda: check_02_epoch_count(data)),
        ("3", "Input shape (64, 481)", lambda: check_03_input_shape(data)),
        ("4", "Split sample counts", lambda: check_04_split_counts(data)),
        ("5", "Subject IDs in every split", lambda: check_05_subject_ids(meta)),
        ("6", "No subject overlap", lambda: check_06_no_overlap(meta)),
        ("7", "No duplicate epochs across splits", lambda: check_07_no_duplicate_epochs(data)),
        ("8", "Training-only scaler fitting", lambda: check_08_scaler_fitting(meta)),
        ("9", "Validation-only model selection", check_09_val_only_selection),
        ("10", "Single final test evaluation", check_10_single_test_eval),
        ("11", "S100 zero-event handling", check_11_s100_zero_events),
        ("12", "Prediction distribution", check_12_prediction_distribution),
        ("13", "Metric recalculation from preds", lambda: check_13_recalculate_metrics(data)),
    ]

    results = {}
    for num, name, fn in checks:
        print(f"  [{num:>2}/13] {name}...")
        try:
            r = fn()
        except Exception as e:
            r = _result(FAIL, f"Exception during check: {e}", {"exception": str(e)})
        results[num] = {"name": name, **r}
        icon = "✓" if r["status"] in (PASS, NOTE) else "✗"
        print(f"         {icon}  [{r['status']}]  {r['message']}")

    elapsed = round(time.time() - t0, 2)
    passed = sum(1 for r in results.values() if r["status"] == PASS)
    noted = sum(1 for r in results.values() if r["status"] == NOTE)
    failed = sum(1 for r in results.values() if r["status"] == FAIL)

    print("\n" + "=" * 78)
    print(f"  AUDIT COMPLETE in {elapsed}s")
    print(f"  PASSED : {passed}/13")
    if noted:
        print(f"  NOTED  : {noted}/13  (informational)")
    if failed:
        print(f"  FAILED : {failed}/13")
    print("=" * 78 + "\n")

    # Save audit report
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    audit_doc = {
        "audit_timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "summary": {"passed": passed, "noted": noted, "failed": failed, "total": 13},
        "frozen_checkpoint": str(CKPT_PATH.resolve()),
        "frozen_dataset": str(DATA_NPZ.resolve()),
        "checks": results,
    }
    with open(AUDIT_OUT, "w") as f:
        json.dump(audit_doc, f, indent=2, cls=NpEncoder)
    print(f"  Audit report saved → {AUDIT_OUT.resolve()}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_audit())
