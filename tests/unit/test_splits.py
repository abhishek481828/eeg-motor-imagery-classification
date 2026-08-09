"""Unit tests for subject-independent dataset splitting (Phase 4)."""

import json
from pathlib import Path

from eeg_mi.data.splits import generate_subject_splits, save_split_manifest


def test_subject_split_disjoint() -> None:
    """Test that train, validation, and test subject splits are strictly disjoint."""
    subjects = list(range(1, 110))
    manifest = generate_subject_splits(
        subjects, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    train_set = set(manifest["splits"]["train"])
    val_set = set(manifest["splits"]["validation"])
    test_set = set(manifest["splits"]["test"])

    # Prove zero overlap (data leakage prevention)
    assert train_set.isdisjoint(val_set), "Train and Val subjects overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test subjects overlap!"
    assert val_set.isdisjoint(test_set), "Val and Test subjects overlap!"


def test_subject_split_coverage() -> None:
    """Test that all subjects are assigned across splits without loss."""
    subjects = list(range(1, 101))
    manifest = generate_subject_splits(subjects, seed=123)

    train = manifest["splits"]["train"]
    val = manifest["splits"]["validation"]
    test = manifest["splits"]["test"]

    total_split_count = len(train) + len(val) + len(test)
    assert total_split_count == len(subjects)


def test_save_split_manifest(tmp_path: Path) -> None:
    """Test saving and verifying JSON split manifest content."""
    subjects = list(range(1, 21))
    manifest = generate_subject_splits(subjects, seed=99)

    out_json = tmp_path / "subject_split.json"
    save_split_manifest(manifest, out_json)

    assert out_json.exists()
    with open(out_json) as f:
        loaded = json.load(f)

    assert loaded["random_seed"] == 99
    assert loaded["num_subjects_total"] == 20
    assert "creation_timestamp" in loaded
    assert "dataset_version" in loaded
    assert len(loaded["splits"]["train"]) > 0
    assert len(loaded["splits"]["validation"]) > 0
    assert len(loaded["splits"]["test"]) > 0
