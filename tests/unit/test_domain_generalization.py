"""Unit tests for domain generalization losses, scalers, channel selection, and augmentations."""

import numpy as np
import torch

from eeg_mi.domain_generalization.augmentations import DomainGeneralizationAugmenter
from eeg_mi.domain_generalization.channel_selection import ChannelSelector
from eeg_mi.domain_generalization.losses import CORALLoss, MMDLoss
from eeg_mi.domain_generalization.normalization import PerChannelZScoreScaler, SubjectRobustScaler


def test_coral_loss_zero() -> None:
    loss_fn = CORALLoss()
    feat = torch.randn(8, 16)
    loss = loss_fn(feat, feat)
    assert torch.abs(loss) < 1e-4


def test_mmd_loss_zero() -> None:
    loss_fn = MMDLoss()
    feat = torch.randn(8, 16)
    loss = loss_fn(feat, feat)
    assert torch.abs(loss) < 1e-4


def test_per_channel_zscore_scaler() -> None:
    X_tr = np.random.randn(10, 64, 480).astype(np.float32)
    scaler = PerChannelZScoreScaler().fit(X_tr)
    X_norm = scaler.transform(X_tr)

    assert X_norm.shape == X_tr.shape
    # Check mean per channel is close to 0
    means = np.mean(X_norm, axis=(0, 2))
    assert np.allclose(means, 0.0, atol=1e-2)


def test_subject_robust_scaler() -> None:
    X_tr = np.random.randn(10, 64, 480).astype(np.float32)
    scaler = SubjectRobustScaler().fit(X_tr)
    X_norm = scaler.transform(X_tr)
    assert X_norm.shape == X_tr.shape


def test_channel_selector_modes() -> None:
    X_tr = np.random.randn(10, 64, 480).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=10)

    cs_all = ChannelSelector(mode="all").fit(X_tr, y_tr)
    assert len(cs_all.selected_indices) == 64

    cs_motor = ChannelSelector(mode="motor_cortex").fit(X_tr, y_tr)
    assert len(cs_motor.selected_indices) == 21

    cs_mi = ChannelSelector(mode="mutual_info", k_channels=16).fit(X_tr, y_tr)
    assert len(cs_mi.selected_indices) == 16


def test_domain_generalization_augmenter() -> None:
    aug = DomainGeneralizationAugmenter(apply_p=1.0)
    aug.train()
    x = torch.randn(4, 64, 480)
    x_aug = aug(x)
    assert x_aug.shape == x.shape
