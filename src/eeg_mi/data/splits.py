"""Subject-independent train/val/test splitting without data leakage."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def generate_subject_splits(
    subject_ids: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    dataset_version: str = "1.0.0",
) -> dict[str, Any]:
    """Generate subject-independent split manifest. Ensures disjoint sets."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    unique_subjects = sorted(set(subject_ids))
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_subjects = sorted(shuffled[:n_train])
    val_subjects = sorted(shuffled[n_train : n_train + n_val])
    test_subjects = sorted(shuffled[n_train + n_val :])

    # Validation: strict disjoint property
    assert set(train_subjects).isdisjoint(set(val_subjects))
    assert set(train_subjects).isdisjoint(set(test_subjects))
    assert set(val_subjects).isdisjoint(set(test_subjects))

    manifest = {
        "dataset_version": dataset_version,
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "random_seed": seed,
        "num_subjects_total": n_total,
        "splits": {
            "train": train_subjects,
            "validation": val_subjects,
            "test": test_subjects,
        },
    }
    return manifest


def save_split_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Save split manifest to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
