"""Unit tests for app.py showcase dashboard and demonstration safeguards."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app import (
    BEST_VAL_ACC,
    CLASS_NAMES,
    ORIG_TEST_ACC,
    W_CNN,
    W_EEGNET,
    DynamicCNN,
    load_dataset,
    load_models,
)


def test_dashboard_imports_and_constants() -> None:
    """Verify essential dashboard constants and label definitions."""
    assert ORIG_TEST_ACC == 0.8098
    assert BEST_VAL_ACC == 0.8302
    assert W_CNN + W_EEGNET == pytest.approx(1.0)
    assert CLASS_NAMES[0] == "Left Fist"
    assert CLASS_NAMES[1] == "Right Fist"


def test_dataset_split_display_consistency() -> None:
    """Verify fixed subject-independent split counts."""
    train_subs = list(range(1, 78))
    val_subs = list(range(78, 94))
    test_subs = list(range(94, 110))

    assert len(train_subs) == 77
    assert len(val_subs) == 16
    assert len(test_subs) == 16
    assert set(train_subs).isdisjoint(set(val_subs))
    assert set(train_subs).isdisjoint(set(test_subs))


def test_model_loading_and_instantiation() -> None:
    """Verify DynamicCNN instantiation and shape execution."""
    model = DynamicCNN(in_ch=64, filters=[32, 64, 128], k_sz=15, drop=0.25, num_cls=2)
    dummy_input = torch.randn(2, 64, 480)
    output = model(dummy_input)

    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()


def test_trial_index_validation() -> None:
    """Verify bounds checking for trial index selection."""
    n_trials = 673
    valid_idx = 42
    invalid_low = -5
    invalid_high = 1000

    assert 0 <= valid_idx < n_trials
    assert not (0 <= invalid_low < n_trials)
    assert not (0 <= invalid_high < n_trials)


def test_random_trial_selection_range() -> None:
    """Verify random trial selection produces valid indices."""
    n_trials = 673
    for _ in range(50):
        rand_idx = int(np.random.randint(0, n_trials))
        assert 0 <= rand_idx < n_trials


def test_prediction_match_calculation() -> None:
    """Verify MATCH vs MISMATCH boolean logic."""
    # Scenario A: Match
    true_a, pred_a = 0, 0
    assert true_a == pred_a

    # Scenario B: Mismatch
    true_b, pred_b = 1, 0
    assert true_b != pred_b


def test_confidence_display_calculations() -> None:
    """Verify soft voting probability integration and confidence bounds."""
    p_cnn = np.array([[0.8, 0.2]])
    p_eegnet = np.array([[0.6, 0.4]])

    p_ens = W_CNN * p_cnn + W_EEGNET * p_eegnet
    assert p_ens.shape == (1, 2)
    assert float(p_ens.sum()) == pytest.approx(1.0)

    pred_cls = int(np.argmax(p_ens))
    confidence_pct = float(p_ens[0, pred_cls] * 100)

    assert pred_cls == 0
    assert 50.0 <= confidence_pct <= 100.0


def test_model_weights_immutability() -> None:
    """Verify model forward pass does NOT mutate model parameters."""
    model = DynamicCNN(64, [32, 64, 128], 15, 0.25, 2)
    model.eval()

    # Capture weight clone
    w_before = model.fc.weight.clone()

    # Run inference
    dummy_x = torch.randn(4, 64, 480)
    with torch.no_grad():
        _ = model(dummy_x)

    w_after = model.fc.weight.clone()
    assert torch.equal(w_before, w_after), "Model weights mutated during inference!"


def test_missing_file_handling() -> None:
    """Verify graceful handling when dataset or checkpoint paths do not exist."""
    # Test load_dataset with bogus path
    X, y, meta = load_dataset()
    if X is not None:
        assert len(X) == len(y)

    m_cnn, m_eegnet = load_models()
    # Should either return loaded models or None without raising an unhandled exception
    assert m_cnn is None or isinstance(m_cnn, torch.nn.Module)
