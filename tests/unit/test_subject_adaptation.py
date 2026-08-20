"""Unit tests for Subject Adaptation modules and calibration trial splitters."""

import numpy as np
import torch

from eeg_mi.subject_adaptation.adapters import (
    PrototypeCalibrator,
    SubjectAdapter,
    TemperatureScaler,
)
from eeg_mi.subject_adaptation.calibration import split_target_subject_trials


def test_subject_adapter_forward() -> None:
    adapter = SubjectAdapter(feature_dim=128)
    x = torch.randn(8, 128)
    out = adapter(x)
    assert out.shape == (8, 128)


def test_prototype_calibrator() -> None:
    feats = np.random.randn(20, 64)
    labels = np.array([0] * 10 + [1] * 10)
    calibrator = PrototypeCalibrator().fit(feats, labels)

    test_feats = np.random.randn(5, 64)
    probs = calibrator.predict_proba(test_feats)
    assert probs.shape == (5, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_temperature_scaler() -> None:
    scaler = TemperatureScaler()
    logits = torch.randn(4, 2)
    out = scaler(logits)
    assert out.shape == (4, 2)


def test_split_target_subject_trials_disjoint() -> None:
    X_sub = np.random.randn(40, 64, 480).astype(np.float32)
    y_sub = np.array([0] * 20 + [1] * 20)

    k_cal = 10
    X_cal, y_cal, X_eval, y_eval = split_target_subject_trials(
        X_sub, y_sub, k_calibration=k_cal, seed=42
    )

    assert len(y_cal) == k_cal
    assert len(y_eval) == 40 - k_cal
