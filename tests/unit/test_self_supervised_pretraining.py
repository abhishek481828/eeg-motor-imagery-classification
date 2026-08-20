"""Unit tests for Self-Supervised Pretraining modules and pretext loss functions."""

import numpy as np
import torch

from eeg_mi.self_supervised.pretraining import ContrastiveNTXentLoss, SelfSupervisedPretrainer


def test_ntxent_loss() -> None:
    loss_fn = ContrastiveNTXentLoss(temperature=0.1)
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0.0


def test_pretrainer_instantiation() -> None:
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64 * 480, 2)

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.size(0), 128 * 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.view(x.size(0), -1))

    enc = DummyEncoder()
    pretrainer = SelfSupervisedPretrainer(enc, pretext_task="contrastive")
    X_tr = np.random.randn(8, 64, 480).astype(np.float32)
    out_enc = pretrainer.pretrain(X_tr, epochs=1)
    assert out_enc is not None
