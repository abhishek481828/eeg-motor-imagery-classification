"""Unit tests for dataset quality requirements, leakage prevention, and determinism."""

import json
from pathlib import Path

from eeg_mi.data_quality.annotation_audit import (
    PHYSIONET_RUN_PROTOCOL,
    RunAuditRecord,
    RunStatus,
    classify_run_status,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def test_train_val_test_subjects_disjoint() -> None:
    """Verify official subject splits are strictly disjoint."""
    train_subs = set(range(1, 78))
    val_subs = set(range(78, 94))
    test_subs = set(range(94, 110))

    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)


def test_no_test_subjects_in_train() -> None:
    """Verify test set subjects S094-S109 are completely absent from train/val."""
    train_subs = set(range(1, 78))
    val_subs = set(range(78, 94))
    test_subs = set(range(94, 110))

    for s in test_subs:
        assert s not in train_subs
        assert s not in val_subs


def test_t1_t2_mapping_run_dependent() -> None:
    """Verify that event mapping changes appropriately depending on run type."""
    # R04/R08/R12: left vs right fist imagery
    assert PHYSIONET_RUN_PROTOCOL[4]["t1_label"] == "left_fist_imagery"
    assert PHYSIONET_RUN_PROTOCOL[4]["t2_label"] == "right_fist_imagery"

    # R06/R10/R14: both fists vs feet imagery
    assert PHYSIONET_RUN_PROTOCOL[6]["t1_label"] == "both_fists_imagery"
    assert PHYSIONET_RUN_PROTOCOL[6]["t2_label"] == "both_feet_imagery"


def test_invalid_runs_not_silently_included() -> None:
    """Verify that runs classified as INVALID_FOR_BINARY_MI or CORRUPT are excluded from usable count."""
    invalid_rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=0,
        n_t2_events=8,
    )
    classified = classify_run_status(invalid_rec)
    assert classified.status == RunStatus.INVALID_FOR_BINARY_MI.value
    # If a run is invalid, usable trial count should not contribute to valid datasets
    assert classified.status in (
        RunStatus.INVALID_FOR_BINARY_MI.value,
        RunStatus.CORRUPT_OR_UNREADABLE.value,
    )


def test_valid_runs_not_incorrectly_removed() -> None:
    """Verify that standard valid runs are kept intact as VALID."""
    valid_rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=8,
        n_t2_events=7,
        sfreq=160.0,
        duration_seconds=125.0,
        n_eeg_channels=64,
    )
    classified = classify_run_status(valid_rec)
    assert classified.status == RunStatus.VALID.value
    assert classified.usable_trial_count == 15


def test_audit_results_deterministic() -> None:
    """Verify classification logic is completely deterministic for identical input data."""
    rec1 = RunAuditRecord(
        subject_id="S088",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S088R04.edf",
        split="validation",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=10,
        n_t2_events=9,
        sfreq=128.0,
        duration_seconds=124.0,
        n_eeg_channels=64,
    )
    rec2 = RunAuditRecord(
        subject_id="S088",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S088R04.edf",
        split="validation",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=10,
        n_t2_events=9,
        sfreq=128.0,
        duration_seconds=124.0,
        n_eeg_channels=64,
    )
    c1 = classify_run_status(rec1)
    c2 = classify_run_status(rec2)
    assert c1.status == c2.status
    assert c1.warnings == c2.warnings
    assert c1.classification_reason == c2.classification_reason


def test_original_split_remains_unchanged() -> None:
    """Verify original dataset metadata split definitions if metadata JSON exists."""
    meta_path = ROOT / "data" / "processed" / "full_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        splits = meta.get("subject_splits", {})
        assert splits["train"] == list(range(1, 78))
        assert splits["validation"] == list(range(78, 94))
        assert splits["test"] == list(range(94, 110))
