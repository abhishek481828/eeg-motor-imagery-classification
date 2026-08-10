#!/usr/bin/env python3
"""Preprocessing Script for Full PhysioNet EEG Dataset (109 Subjects, 327 EDF Files).

Processes motor imagery runs R04, R08, R12 across subjects S001..S109.
Maps T1 -> Class 0 (Left Fist), T2 -> Class 1 (Right Fist), ignores T0.
Segments event-onset windows (shape 64x480).
Fits TrainFittedScaler strictly on training subjects S001..S077.
Saves data/processed/full_dataset.npz and data/processed/full_metadata.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from eeg_mi.data.splits import save_split_manifest
from eeg_mi.preprocessing.normalization import TrainFittedScaler
from eeg_mi.preprocessing.pipeline import PreprocessingPipeline, extract_run_id_from_filename
from eeg_mi.utils.logging import get_logger

logger = get_logger("PreprocessFullDataset")


def run_full_preprocessing(
    raw_dir: Path = Path("data/raw/physionet"),
    processed_dir: Path = Path("data/processed"),
    split_manifest_path: Path = Path("data/splits/full_subject_split.json"),
    out_npz_name: str = "full_dataset.npz",
    out_meta_name: str = "full_metadata.json",
) -> int:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    allowed_runs = [4, 8, 12]

    # Verify raw EDF files exist
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory '{raw_dir.resolve()}' not found!")

    edf_files = sorted(list(raw_dir.glob("**/*.edf")) + list(raw_dir.glob("**/*.EDF")))
    target_edf_files = [
        f for f in edf_files if extract_run_id_from_filename(f.name) in allowed_runs
    ]

    if not target_edf_files:
        raise FileNotFoundError(f"No EDF files for allowed runs {allowed_runs} found in {raw_dir}")

    # Discover subjects S001..S109
    discovered_subjects = sorted(
        {int(f.parent.name[1:]) for f in target_edf_files if f.parent.name.startswith("S")}
    )

    # Fixed Subject-Independent Split for 109 subjects:
    # Train: 77 subjects (S001-S077)
    # Val  : 16 subjects (S078-S093)
    # Test : 16 subjects (S094-S109)
    train_subs = [s for s in discovered_subjects if 1 <= s <= 77]
    val_subs = [s for s in discovered_subjects if 78 <= s <= 93]
    test_subs = [s for s in discovered_subjects if 94 <= s <= 109]

    # Verify disjoint splits
    assert set(train_subs).isdisjoint(set(val_subs)), "Train and Val subjects overlap!"
    assert set(train_subs).isdisjoint(set(test_subs)), "Train and Test subjects overlap!"
    assert set(val_subs).isdisjoint(set(test_subs)), "Val and Test subjects overlap!"

    split_manifest = {
        "dataset_version": "1.0.0",
        "random_seed": 42,
        "num_subjects_total": len(discovered_subjects),
        "splits": {
            "train": train_subs,
            "validation": val_subs,
            "test": test_subs,
        },
    }
    save_split_manifest(split_manifest, split_manifest_path)

    pipeline = PreprocessingPipeline(
        l_freq=7.0,
        h_freq=30.0,
        notch_freq=60.0,
        window_duration=3.0,
        allowed_runs=allowed_runs,
    )

    logger.info(
        f"Preprocessing {len(target_edf_files)} EDF files across {len(discovered_subjects)} subjects..."
    )

    train_windows, train_labels, train_meta = [], [], []
    val_windows, val_labels, val_meta = [], [], []
    test_windows, test_labels, test_meta = [], [], []

    rejected_epochs_count = 0
    rejected_files_count = 0
    per_recording_counts: dict[str, int] = {}
    per_subject_counts: dict[str, int] = {}

    for idx, fpath in enumerate(target_edf_files, 1):
        sub_id_int = int(fpath.parent.name[1:]) if fpath.parent.name.startswith("S") else 0
        sub_str = fpath.parent.name
        rec_str = fpath.stem

        if idx % 50 == 0 or idx == len(target_edf_files):
            print(f"[{idx}/{len(target_edf_files)}] Processing {fpath.name}...")

        try:
            wins, lbls, metas = pipeline.process_recording(fpath)
            if len(wins) == 0:
                rejected_files_count += 1
                continue

            valid_w, valid_l, valid_m = [], [], []
            for w, target_lbl, m in zip(wins, lbls, metas, strict=False):
                # Pipeline returns 481 samples (inclusive endpoint: 160 Hz × 3.0 s + 1)
                if (
                    w.shape[0] == 64
                    and w.shape[1] in (480, 481)
                    and int(target_lbl) in (0, 1)
                    and not np.isnan(w).any()
                    and not np.isinf(w).any()
                ):
                    valid_w.append(w)
                    valid_l.append(target_lbl)
                    valid_m.append(m)
                else:
                    rejected_epochs_count += 1

            if not valid_w:
                rejected_files_count += 1
                continue

            valid_w = np.array(valid_w)
            valid_l = np.array(valid_l)

            per_recording_counts[rec_str] = len(valid_w)
            per_subject_counts[sub_str] = per_subject_counts.get(sub_str, 0) + len(valid_w)

            if sub_id_int in train_subs:
                train_windows.append(valid_w)
                train_labels.append(valid_l)
                train_meta.extend(valid_m)
            elif sub_id_int in val_subs:
                val_windows.append(valid_w)
                val_labels.append(valid_l)
                val_meta.extend(valid_m)
            elif sub_id_int in test_subs:
                test_windows.append(valid_w)
                test_labels.append(valid_l)
                test_meta.extend(valid_m)
        except Exception as e:
            logger.error(f"Failed to process {fpath.name}: {e}")
            rejected_files_count += 1

    if not train_windows:
        raise ValueError("No valid training epochs extracted from EDF files!")

    X_train = np.concatenate(train_windows, axis=0).astype(np.float32)
    y_train = np.concatenate(train_labels, axis=0).astype(np.int64)

    # Determine actual time dimension from first training window
    _time_dim = (
        X_train.shape[2]
        if len(X_train.shape) == 3
        else train_windows[0].shape[1]
        if False
        else X_train.shape[2]
    )
    X_val = (
        np.concatenate(val_windows, axis=0).astype(np.float32)
        if val_windows
        else np.empty((0, 64, _time_dim), dtype=np.float32)
    )
    y_val = (
        np.concatenate(val_labels, axis=0).astype(np.int64)
        if val_labels
        else np.empty((0,), dtype=np.int64)
    )

    X_test = (
        np.concatenate(test_windows, axis=0).astype(np.float32)
        if test_windows
        else np.empty((0, 64, _time_dim), dtype=np.float32)
    )
    y_test = (
        np.concatenate(test_labels, axis=0).astype(np.int64)
        if test_labels
        else np.empty((0,), dtype=np.int64)
    )

    # Zero-Leakage Normalization: Fit scaler on training subjects ONLY
    logger.info("Fitting TrainFittedScaler strictly on training subject windows...")
    scaler = TrainFittedScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val) if len(X_val) > 0 else X_val
    X_test_norm = scaler.transform(X_test) if len(X_test) > 0 else X_test

    out_npz = processed_dir / out_npz_name
    np.savez_compressed(
        out_npz,
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
    )

    out_meta = processed_dir / out_meta_name
    all_metadata_combined = train_meta + val_meta + test_meta
    with open(out_meta, "w") as f:
        json.dump(
            {
                "config_hash": pipeline.config_hash,
                "num_channels": 64,
                "sampling_rate": 160,
                "epoch_shape": [64, _time_dim],
                "allowed_runs": allowed_runs,
                "subject_splits": split_manifest["splits"],
                "epochs_count": {
                    "train": len(X_train),
                    "val": len(X_val),
                    "test": len(X_test),
                    "total": len(X_train) + len(X_val) + len(X_test),
                },
                "per_subject_epochs": per_subject_counts,
                "per_recording_epochs": per_recording_counts,
                "records_metadata": all_metadata_combined,
            },
            f,
            indent=2,
        )

    total_epochs = len(X_train) + len(X_val) + len(X_test)
    all_y = np.concatenate([y_train, y_val, y_test], axis=0)
    cls_0_count = int(np.sum(all_y == 0))
    cls_1_count = int(np.sum(all_y == 1))

    print("\n" + "=" * 75)
    print("      FULL PHYSIOET EEG DATASET PREPROCESSING SUMMARY")
    print("=" * 75)
    print(f"EDF Files Processed: {len(target_edf_files)}")
    print(f"Total Subjects     : {len(discovered_subjects)} (S001 - S109)")
    print(f"Allowed Runs       : {allowed_runs} (2-Class Left vs Right Fist)")
    print("-" * 75)
    print(f"Total Epochs       : {total_epochs}")
    print(f"  - Class 0 (Left Fist)  : {cls_0_count} epochs")
    print(f"  - Class 1 (Right Fist) : {cls_1_count} epochs")
    print("-" * 75)
    print(f"Train Epochs ({len(train_subs)} subs) : {len(X_train)} samples")
    print(f"Val Epochs   ({len(val_subs)} subs) : {len(X_val)} samples")
    print(f"Test Epochs  ({len(test_subs)} subs) : {len(X_test)} samples")
    print("-" * 75)
    print(f"Epoch Tensor Shape : (N, 64, {_time_dim})")
    print(f"Rejected Epochs    : {rejected_epochs_count}")
    print(f"Rejected Files     : {rejected_files_count}")
    print(f"Saved Processed NPZ: {out_npz.resolve()}")
    print(f"Saved Metadata JSON: {out_meta.resolve()}")
    print("=" * 75 + "\n")

    return 0


def main() -> int:
    return run_full_preprocessing()


if __name__ == "__main__":
    sys.exit(main())
