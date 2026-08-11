"""Unit tests for Accuracy Improvement Study models and feature fusion architectures."""

import torch

from eeg_mi.models.factory import count_parameters, create_model
from eeg_mi.models.multiscale_cnn import MultiScaleCNN
from eeg_mi.models.riemannian_fusion import CNNRiemannianFusionModel
from eeg_mi.models.temporal_attention import TemporalAttentionCNN


def test_multiscale_cnn_shape() -> None:
    model = MultiScaleCNN(in_channels=64, num_classes=2)
    x = torch.randn(4, 64, 480)
    out = model(x)
    assert out.shape == (4, 2)


def test_temporal_attention_cnn_shape() -> None:
    model = TemporalAttentionCNN(in_channels=64, num_classes=2)
    x = torch.randn(4, 64, 480)
    out = model(x)
    assert out.shape == (4, 2)


def test_riemannian_fusion_shape() -> None:
    model = CNNRiemannianFusionModel(in_channels=64, riemannian_dim=2080, num_classes=2)
    x_signal = torch.randn(4, 64, 480)
    x_riem = torch.randn(4, 2080)
    out = model(x_signal, x_riem)
    assert out.shape == (4, 2)


def test_factory_instantiation() -> None:
    m1 = create_model("multiscale_cnn", num_channels=64, num_classes=2)
    m2 = create_model("temporal_attention", num_channels=64, num_classes=2)
    m3 = create_model("riemannian_fusion", num_channels=64, riemannian_dim=100, num_classes=2)

    assert isinstance(m1, MultiScaleCNN)
    assert isinstance(m2, TemporalAttentionCNN)
    assert isinstance(m3, CNNRiemannianFusionModel)
    assert count_parameters(m1)[0] > 0
