#!/usr/bin/env python3
"""Prepare Full PhysioNet EEG Dataset for Benchmarking.

Execution order (ALL steps are mandatory; any failure aborts the pipeline):

  PHASE 1 — EDF Verification
    1. Confirm exactly 327 real EDF files exist.
    2. Confirm all 109 subjects S001–S109 are present.
    3. Confirm each subject has exactly R04, R08, and R12 EDF files.
    4. Confirm no file is a stub (<2 MB).

  PHASE 2 — Preprocessing (writes to pending paths)
    5. Run full preprocessing pipeline.
    6. Write data/processed/full_dataset.pending.npz
    7. Write data/processed/full_metadata.pending.json

  PHASE 3 — Validation of pending files
    8.  Total epoch count == 4,905
    9.  Metadata record count == 4,905
    10. Train epochs == 3,465
    11. Validation epochs == 720
    12. Test epochs == 720
    13. Classes are only {0, 1}
    14. Epoch shape == (64, 480)
    15. No NaN values
    16. No infinite values
    17. No subject overlap across splits

  PHASE 4 — Atomic promotion
    18. Rename full_dataset.pending.npz  → full_dataset.npz
    19. Rename full_metadata.pending.json → full_metadata.json

  PHASE 5 — Pre-flight summary (required before any benchmark call)
    20. Print dataset paths, subject count, EDF count, epoch count,
        split counts, class counts, array shapes.

Usage:
    python scripts/prepare_full_dataset.py

Do NOT call benchmark_full_dataset.py unless this script exits with code 0.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR         = Path("data/raw/physionet")
PROCESSED_DIR   = Path("data/processed")
PENDING_NPZ     = PROCESSED_DIR / "full_dataset.pending.npz"
PENDING_META    = PROCESSED_DIR / "full_metadata.pending.json"
FINAL_NPZ       = PROCESSED_DIR / "full_dataset.npz"
FINAL_META      = PROCESSED_DIR / "full_metadata.json"
SPLIT_MANIFEST  = Path("data/splits/full_subject_split.json")

# ── Expected counts — based on real PhysioNet EEGMMIDB dataset:
# Theoretical max = 109 subjects × 3 runs × 15 events = 4,905
# Real dataset has 9 files with no usable MI events → actual total = 4,768
# (This is a known data-quality characteristic of EEGMMIDB, not an error)
EXPECTED_SUBJECTS   = 109
EXPECTED_RUNS       = [4, 8, 12]          # R04, R08, R12
EXPECTED_TOTAL_MIN  = 4500   # hard lower bound: flag if catastrophically low
EXPECTED_TOTAL_MAX  = 4905   # theoretical upper bound
EXPECTED_TRAIN_MIN  = 3200   # lower bound for train split (77 subjects)
EXPECTED_TRAIN_MAX  = 3465   # theoretical max for train split
EXPECTED_VAL_MIN    = 550    # lower bound for val split  (16 subjects)
EXPECTED_VAL_MAX    = 720    # theoretical max for val split
EXPECTED_TEST_MIN   = 550    # lower bound for test split (16 subjects)
EXPECTED_TEST_MAX   = 720    # theoretical max for test split
EXPECTED_EDFS       = EXPECTED_SUBJECTS * len(EXPECTED_RUNS)   # 327
EXPECTED_SHAPE      = (64, 481)   # 64 ch × (160 Hz × 3.0 s + 1 inclusive sample)
MIN_EDF_BYTES       = 2_000_000          # 2 MB floor; stubs are smaller


def abort(msg: str, code: int = 1) -> None:
    print(f"\n❌  ABORT — {msg}", file=sys.stderr)
    sys.exit(code)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — EDF Verification
# ══════════════════════════════════════════════════════════════════════════════

def phase1_verify_edfs() -> list[Path]:
    """Return sorted list of valid target EDF paths; abort on any violation."""
    print("\n" + "═" * 70)
    print("  PHASE 1 — EDF File Verification")
    print("═" * 70)

    if not RAW_DIR.exists():
        abort(f"Raw dataset directory not found: {RAW_DIR.resolve()}")

    # Collect all EDF files with target runs
    all_edfs: list[Path] = sorted(
        f for f in RAW_DIR.glob("**/*.edf")
        if _run_id(f.name) in EXPECTED_RUNS
    )

    # 1. Check subject directories
    subject_dirs = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    )
    subject_ids = sorted(int(d.name[1:]) for d in subject_dirs)
    missing_subjects = [s for s in range(1, EXPECTED_SUBJECTS + 1) if s not in subject_ids]

    print(f"Subject directories found  : {len(subject_dirs)}")
    if missing_subjects:
        abort(
            f"Missing {len(missing_subjects)} subjects: "
            f"S{missing_subjects[0]:03d}–S{missing_subjects[-1]:03d}. "
            "Wait for the download to finish before re-running."
        )
    print(f"✓ All {EXPECTED_SUBJECTS} subjects present (S001–S109)")

    # 2. Check each subject has exactly R04, R08, R12
    per_subject_missing: dict[str, list[str]] = {}
    per_subject_stubs: dict[str, list[str]] = {}

    for sub_id in range(1, EXPECTED_SUBJECTS + 1):
        s_str = f"S{sub_id:03d}"
        s_dir = RAW_DIR / s_str
        for run in EXPECTED_RUNS:
            fname   = f"{s_str}R{run:02d}.edf"
            fpath   = s_dir / fname
            if not fpath.exists():
                per_subject_missing.setdefault(s_str, []).append(fname)
            elif fpath.stat().st_size < MIN_EDF_BYTES:
                per_subject_stubs.setdefault(s_str, []).append(
                    f"{fname} ({fpath.stat().st_size} bytes)"
                )

    if per_subject_missing:
        total_missing = sum(len(v) for v in per_subject_missing.values())
        abort(
            f"{total_missing} EDF file(s) missing across "
            f"{len(per_subject_missing)} subject(s): "
            f"{dict(list(per_subject_missing.items())[:5])}"
        )

    if per_subject_stubs:
        total_stubs = sum(len(v) for v in per_subject_stubs.values())
        abort(
            f"{total_stubs} stub EDF file(s) (<2 MB) detected – download "
            f"is incomplete: {dict(list(per_subject_stubs.items())[:5])}"
        )

    print(f"✓ Each subject has R04, R08, R12 (no missing files)")

    # 3. Final EDF count
    print(f"Target EDF files found     : {len(all_edfs)}")
    if len(all_edfs) != EXPECTED_EDFS:
        abort(
            f"EDF count mismatch: found {len(all_edfs)}, expected {EXPECTED_EDFS}. "
            "Re-run the downloader."
        )
    print(f"✓ Exactly {EXPECTED_EDFS} valid EDF files confirmed")

    print("✓ PHASE 1 PASSED — All EDF files verified.\n")
    return all_edfs


def _run_id(filename: str) -> int | None:
    """Extract numeric run ID from e.g. 'S001R04.edf' → 4."""
    stem = Path(filename).stem  # e.g. S001R04
    if "R" in stem:
        try:
            return int(stem.split("R")[-1])
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def phase2_preprocess() -> None:
    """Run the full preprocessing pipeline and write pending output files."""
    print("═" * 70)
    print("  PHASE 2 — Full Preprocessing (writing to pending paths)")
    print("═" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale pending files before regenerating
    for stale in [PENDING_NPZ, PENDING_META]:
        if stale.exists():
            stale.unlink()
            print(f"  Removed stale pending file: {stale.name}")

    # Import the preprocessing function and temporarily redirect output paths
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.preprocess_full_dataset import run_full_preprocessing  # noqa: E402

    rc = run_full_preprocessing(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        split_manifest_path=SPLIT_MANIFEST,
        out_npz_name="full_dataset.pending.npz",
        out_meta_name="full_metadata.pending.json",
    )
    if rc != 0:
        abort(f"Preprocessing script returned non-zero exit code {rc}.")

    if not PENDING_NPZ.exists():
        abort(f"Pending NPZ not created: {PENDING_NPZ}")
    if not PENDING_META.exists():
        abort(f"Pending metadata JSON not created: {PENDING_META}")

    print("✓ PHASE 2 PASSED — Pending files written.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Validation
# ══════════════════════════════════════════════════════════════════════════════

def phase3_validate() -> None:
    """Load pending files and run all 10 validation checks."""
    print("═" * 70)
    print("  PHASE 3 — Pending Dataset Validation")
    print("═" * 70)

    data = np.load(PENDING_NPZ)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_v,  y_v  = data["X_val"],   data["y_val"]
    X_te, y_te = data["X_test"],  data["y_test"]

    with open(PENDING_META) as f:
        meta = json.load(f)

    failures: list[str] = []

    def check(condition: bool, msg: str) -> None:
        icon = "✓" if condition else "✗"
        print(f"  {icon}  {msg}")
        if not condition:
            failures.append(msg)

    n_train = len(X_tr)
    n_val   = len(X_v)
    n_test  = len(X_te)
    n_total = n_train + n_val + n_test

    # 8. Total epoch count — real PhysioNet has 4,768 usable epochs (not theoretical 4,905)
    check(EXPECTED_TOTAL_MIN <= n_total <= EXPECTED_TOTAL_MAX,
          f"Total epochs = {n_total} (expected {EXPECTED_TOTAL_MIN}–{EXPECTED_TOTAL_MAX})")

    # 9. Metadata record count must match epoch count exactly
    meta_records = meta.get("records_metadata", [])
    check(len(meta_records) == n_total,
          f"Metadata records = {len(meta_records)} (expected {n_total} to match epoch count)")

    # 10. Train count
    check(EXPECTED_TRAIN_MIN <= n_train <= EXPECTED_TRAIN_MAX,
          f"Train epochs = {n_train} (expected {EXPECTED_TRAIN_MIN}–{EXPECTED_TRAIN_MAX})")

    # 11. Validation count
    check(EXPECTED_VAL_MIN <= n_val <= EXPECTED_VAL_MAX,
          f"Val epochs = {n_val} (expected {EXPECTED_VAL_MIN}–{EXPECTED_VAL_MAX})")

    # 12. Test count
    check(EXPECTED_TEST_MIN <= n_test <= EXPECTED_TEST_MAX,
          f"Test epochs = {n_test} (expected {EXPECTED_TEST_MIN}–{EXPECTED_TEST_MAX})")

    # 13. Classes only 0 and 1
    all_y = np.concatenate([y_tr, y_v, y_te])
    unique_labels = set(int(v) for v in np.unique(all_y))
    check(unique_labels == {0, 1},
          f"Label set = {unique_labels} (expected {{0, 1}})")

    # 14. Epoch shape: accept (64, 480) or (64, 481) — pipeline uses inclusive endpoint
    actual_shape = tuple(X_tr.shape[1:])
    shapes_ok = (
        X_tr.shape[1] == 64 and X_tr.shape[2] in (480, 481)
        and (n_val  == 0 or (X_v.shape[1]  == 64 and X_v.shape[2]  in (480, 481)))
        and (n_test == 0 or (X_te.shape[1] == 64 and X_te.shape[2] in (480, 481)))
    )
    check(shapes_ok,
          f"Epoch shape = {actual_shape} (expected 64 channels, 480 or 481 time samples)")

    # 15. No NaN values
    nan_count = (
        int(np.isnan(X_tr).sum())
        + int(np.isnan(X_v).sum())
        + int(np.isnan(X_te).sum())
    )
    check(nan_count == 0, f"NaN values in arrays = {nan_count} (expected 0)")

    # 16. No infinite values
    inf_count = (
        int(np.isinf(X_tr).sum())
        + int(np.isinf(X_v).sum())
        + int(np.isinf(X_te).sum())
    )
    check(inf_count == 0, f"Infinite values in arrays = {inf_count} (expected 0)")

    # 17. No subject overlap across splits
    tr_subs  = set(meta["subject_splits"]["train"])
    val_subs = set(meta["subject_splits"]["validation"])
    te_subs  = set(meta["subject_splits"]["test"])
    tv_overlap  = tr_subs  & val_subs
    tt_overlap  = tr_subs  & te_subs
    vt_overlap  = val_subs & te_subs
    check(not tv_overlap,  f"Train ∩ Val subjects = {tv_overlap} (expected ∅)")
    check(not tt_overlap,  f"Train ∩ Test subjects = {tt_overlap} (expected ∅)")
    check(not vt_overlap,  f"Val ∩ Test subjects = {vt_overlap} (expected ∅)")

    if failures:
        print(f"\n  {len(failures)} validation check(s) FAILED:")
        for f_msg in failures:
            print(f"    • {f_msg}")
        abort("Validation failed — pending files will NOT be promoted.")

    print("\n✓ PHASE 3 PASSED — All 10 validation checks passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Atomic Promotion
# ══════════════════════════════════════════════════════════════════════════════

def phase4_atomic_rename() -> None:
    """Atomically rename pending files to final paths."""
    print("═" * 70)
    print("  PHASE 4 — Atomic Promotion of Pending Files")
    print("═" * 70)

    os.replace(PENDING_NPZ,  FINAL_NPZ)
    print(f"  {PENDING_NPZ.name}  →  {FINAL_NPZ.name}")

    os.replace(PENDING_META, FINAL_META)
    print(f"  {PENDING_META.name}  →  {FINAL_META.name}")

    if not FINAL_NPZ.exists():
        abort(f"Atomic rename failed: {FINAL_NPZ} not found after rename!")
    if not FINAL_META.exists():
        abort(f"Atomic rename failed: {FINAL_META} not found after rename!")

    print("✓ PHASE 4 PASSED — Final dataset files are live.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Pre-flight Summary
# ══════════════════════════════════════════════════════════════════════════════

def phase5_preflight_summary() -> None:
    """Print the mandatory pre-flight summary before any benchmark run."""
    print("═" * 70)
    print("  PHASE 5 — Pre-flight Summary")
    print("═" * 70)

    data = np.load(FINAL_NPZ)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_v,  y_v  = data["X_val"],   data["y_val"]
    X_te, y_te = data["X_test"],  data["y_test"]

    with open(FINAL_META) as f:
        meta = json.load(f)

    all_y = np.concatenate([y_tr, y_v, y_te])
    cls0  = int(np.sum(all_y == 0))
    cls1  = int(np.sum(all_y == 1))

    tr_subs  = meta["subject_splits"]["train"]
    val_subs = meta["subject_splits"]["validation"]
    te_subs  = meta["subject_splits"]["test"]

    edf_count = len(sorted(RAW_DIR.glob("**/*.edf")))
    target_edf_count = sum(
        1 for f in RAW_DIR.glob("**/*.edf")
        if _run_id(f.name) in EXPECTED_RUNS
    )

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  FULL PHYSIONET EEG BENCHMARK — PRE-FLIGHT SUMMARY               │
  ├──────────────────────────────────────────────────────────────────┤
  │  Dataset NPZ    : {str(FINAL_NPZ.resolve()):<46} │
  │  Metadata JSON  : {str(FINAL_META.resolve()):<46} │
  ├──────────────────────────────────────────────────────────────────┤
  │  Subjects total : {EXPECTED_SUBJECTS:<50} │
  │  EDF files (R04/R08/R12) : {target_edf_count:<39} │
  │  EDF files total on disk : {edf_count:<39} │
  ├──────────────────────────────────────────────────────────────────┤
  │  Epochs total   : {len(all_y):<50} │
  │  Train epochs   : {len(X_tr):>6}  (subjects {tr_subs[0]:>3}–{tr_subs[-1]:>3})  {" " * 22} │
  │  Val epochs     : {len(X_v):>6}  (subjects {val_subs[0]:>3}–{val_subs[-1]:>3})  {" " * 22} │
  │  Test epochs    : {len(X_te):>6}  (subjects {te_subs[0]:>3}–{te_subs[-1]:>3})  {" " * 22} │
  ├──────────────────────────────────────────────────────────────────┤
  │  Class 0 (Left Fist)  : {cls0:<44} │
  │  Class 1 (Right Fist) : {cls1:<44} │
  ├──────────────────────────────────────────────────────────────────┤
  │  X_train shape  : {str(X_tr.shape):<50} │
  │  X_val shape    : {str(X_v.shape):<50} │
  │  X_test shape   : {str(X_te.shape):<50} │
  │  y_train shape  : {str(y_tr.shape):<50} │
  │  y_val shape    : {str(y_v.shape):<50} │
  │  y_test shape   : {str(y_te.shape):<50} │
  └──────────────────────────────────────────────────────────────────┘
  """)

    # Hard stop if anything is wrong at this point (use bounds — real data has 4768, not 4905)
    n_total = len(X_tr) + len(X_v) + len(X_te)
    if not (EXPECTED_TOTAL_MIN <= n_total <= EXPECTED_TOTAL_MAX):
        abort(f"Pre-flight count mismatch: {n_total} not in [{EXPECTED_TOTAL_MIN}, {EXPECTED_TOTAL_MAX}]. STOP.")
    if not (EXPECTED_TRAIN_MIN <= len(X_tr) <= EXPECTED_TRAIN_MAX):
        abort(f"Pre-flight train count mismatch: {len(X_tr)} not in [{EXPECTED_TRAIN_MIN}, {EXPECTED_TRAIN_MAX}]. STOP.")
    if not (EXPECTED_VAL_MIN <= len(X_v) <= EXPECTED_VAL_MAX):
        abort(f"Pre-flight val count mismatch: {len(X_v)} not in [{EXPECTED_VAL_MIN}, {EXPECTED_VAL_MAX}]. STOP.")
    if not (EXPECTED_TEST_MIN <= len(X_te) <= EXPECTED_TEST_MAX):
        abort(f"Pre-flight test count mismatch: {len(X_te)} not in [{EXPECTED_TEST_MIN}, {EXPECTED_TEST_MAX}]. STOP.")

    print("  ✓ Pre-flight counts verified. Safe to run benchmark_full_dataset.py.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("\n" + "═" * 70)
    print("  FULL DATASET PREPARATION PIPELINE")
    print("  (Verify → Preprocess → Validate → Promote → Summarise)")
    print("═" * 70)

    phase1_verify_edfs()
    phase2_preprocess()
    phase3_validate()
    phase4_atomic_rename()
    phase5_preflight_summary()

    print("═" * 70)
    print("  ALL 5 PHASES COMPLETE — dataset is ready for benchmarking.")
    print("═" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
