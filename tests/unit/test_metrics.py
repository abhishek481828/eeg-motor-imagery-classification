"""Unit tests for metrics calculation."""

import numpy as np

from eeg_mi.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect() -> None:
    """Test metrics calculation with 100% correct predictions."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])

    res = compute_metrics(y_true, y_pred, class_names=["C0", "C1", "C2", "C3"])

    assert res["accuracy"] == 1.0
    assert res["balanced_accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["cohens_kappa"] == 1.0
