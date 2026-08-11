"""Unit tests for leakage prevention, subject split disjointness, and test set isolation."""

import json
from pathlib import Path

from eeg_mi.data.splits import generate_subject_splits

ROOT = Path(__file__).resolve().parent.parent.parent


def test_subject_splits_disjoint() -> None:
    subjects = list(range(1, 110))
    manifest = generate_subject_splits(
        subjects, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    tr_subs = set(manifest["splits"]["train"])
    v_subs = set(manifest["splits"]["validation"])
    te_subs = set(manifest["splits"]["test"])

    assert tr_subs.isdisjoint(v_subs)
    assert tr_subs.isdisjoint(te_subs)
    assert v_subs.isdisjoint(te_subs)


def test_frozen_test_set_isolation() -> None:
    meta_path = ROOT / "data" / "processed" / "full_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        test_subs = set(meta["subject_splits"]["test"])
        val_subs = set(meta["subject_splits"]["validation"])

        # Test subjects S094-S109 must be disjoint from validation subjects
        assert test_subs.isdisjoint(val_subs)
        assert min(test_subs) >= 94
