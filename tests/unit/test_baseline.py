"""Unit tests for baseline ML classifier (Phase 6)."""

import numpy as np

from eeg_mi.models.baseline import BaselineClassifier


def test_baseline_classifier_fit_evaluate(
    dummy_eeg_data: np.ndarray, dummy_targets: np.ndarray
) -> None:
    """Test fitting and evaluating LDA baseline classifier on dummy EEG data."""
    clf = BaselineClassifier(model_type="lda")
    clf.fit(dummy_eeg_data, dummy_targets)

    metrics = clf.evaluate(dummy_eeg_data, dummy_targets)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "cohens_kappa" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0


def test_baseline_logistic_regression(
    dummy_eeg_data: np.ndarray, dummy_targets: np.ndarray
) -> None:
    """Test Logistic Regression baseline model choice."""
    clf = BaselineClassifier(model_type="logistic_regression")
    clf.fit(dummy_eeg_data, dummy_targets)

    metrics = clf.evaluate(dummy_eeg_data, dummy_targets)
    assert metrics["accuracy"] >= 0.0
