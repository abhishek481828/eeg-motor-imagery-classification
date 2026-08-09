#!/usr/bin/env python3
"""Script to Create Deterministic Subject-Independent Train/Val/Test Splits.

Prevents data leakage by assigning complete subjects exclusively to either
Training, Validation, or Testing split.
Saves split manifest to data/splits/subject_split.json.
"""

import argparse
import sys
from pathlib import Path

from eeg_mi.data.splits import generate_subject_splits, save_split_manifest
from eeg_mi.utils.logging import get_logger

logger = get_logger("CreateSubjectSplits")


def extract_subject_id_from_path(file_path: Path) -> int | None:
    """Extract numeric subject ID from path (e.g. S001/S001R01.edf -> 1)."""
    parent_name = file_path.parent.name
    if parent_name.startswith("S") and parent_name[1:].isdigit():
        return int(parent_name[1:])
    stem = file_path.stem
    if stem.startswith("S") and len(stem) >= 4 and stem[1:4].isdigit():
        return int(stem[1:4])
    return None


def discover_subjects(data_dir: Path, fallback_count: int = 109) -> list[int]:
    """Discover available subject IDs in dataset directory, or fallback to [1..109]."""
    data_dir = Path(data_dir)
    if data_dir.exists():
        edf_files = list(data_dir.glob("**/*.edf")) + list(data_dir.glob("**/*.EDF"))
        subjects = set()
        for f in edf_files:
            sub_id = extract_subject_id_from_path(f)
            if sub_id is not None:
                subjects.add(sub_id)
        if subjects:
            logger.info(f"Discovered {len(subjects)} subjects in '{data_dir}'")
            return sorted(subjects)

    logger.info(f"Using default PhysioNet subject range 1 to {fallback_count}")
    return list(range(1, fallback_count + 1))


def run_split_creation(
    data_dir: str = "data/raw/physionet",
    output: str = "data/splits/subject_split.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> int:
    subject_ids = discover_subjects(Path(data_dir))
    manifest = generate_subject_splits(
        subject_ids=subject_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    output_path = Path(output)
    save_split_manifest(manifest, output_path)

    print("=" * 70)
    print("      Subject-Independent Split Manifest Created Successfully")
    print("=" * 70)
    print(f"Manifest Path     : {output_path.resolve()}")
    print(f"Random Seed       : {manifest['random_seed']}")
    print(f"Total Subjects    : {manifest['num_subjects_total']}")
    print(
        f"Train Subjects ({len(manifest['splits']['train'])}) : {manifest['splits']['train'][:10]}..."
    )
    print(
        f"Val Subjects   ({len(manifest['splits']['validation'])}) : {manifest['splits']['validation'][:5]}..."
    )
    print(
        f"Test Subjects  ({len(manifest['splits']['test'])}) : {manifest['splits']['test'][:5]}..."
    )
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Subject-Independent Dataset Split Manifest"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/physionet",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits/subject_split.json",
        help="Target output split manifest JSON path",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train subject fraction")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation subject fraction")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test subject fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    return run_split_creation(
        data_dir=args.data_dir,
        output=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
