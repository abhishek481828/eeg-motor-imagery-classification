"""Self-Supervised Pretraining Modules for EEG Representation Learning.

Pretrains feature encoders strictly on unlabeled training subject signals (S001-S077)
via masked reconstruction and NT-Xent contrastive learning objectives.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.domain_generalization.augmentations import DomainGeneralizationAugmenter


class ContrastiveNTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross Entropy (NT-Xent) Loss for SimCLR Contrastive Learning."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss between two augmented views z_i and z_j of shape (batch, projection_dim)."""
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Mask out self-contrastive similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        # Ground truth positive targets: view i matches view i+N
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z_i.device),
                torch.arange(0, batch_size, device=z_i.device),
            ]
        )

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class SelfSupervisedPretrainer:
    """Manager for self-supervised pretraining tasks on unlabeled EEG training signals."""

    def __init__(self, encoder: nn.Module, pretext_task: str = "contrastive", lr: float = 0.001):
        self.encoder = encoder
        self.pretext_task = pretext_task.lower()
        self.lr = lr

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(128 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

        # Reconstruction head for masked reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 480),
        )

    def pretrain(
        self,
        X_tr_unlabeled: np.ndarray,
        epochs: int = 15,
        device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        """Pretrain encoder strictly on unlabeled training dataset X_tr_unlabeled of shape (N, C, T)."""
        # Dummy labels since self-supervised pretraining uses NO labels
        dummy_y = np.zeros(len(X_tr_unlabeled), dtype=np.int64)
        loader = DataLoader(EEGDataset(X_tr_unlabeled, dummy_y), batch_size=32, shuffle=True)

        self.encoder.to(device)
        self.projection_head.to(device)
        self.decoder.to(device)

        if self.pretext_task == "contrastive":
            augmenter = DomainGeneralizationAugmenter(apply_p=1.0)
            loss_fn = ContrastiveNTXentLoss(temperature=0.1)
            params = list(self.encoder.parameters()) + list(self.projection_head.parameters())
            opt = torch.optim.Adam(params, lr=self.lr)

            for _epoch in range(epochs):
                self.encoder.train()
                for xb, _ in loader:
                    opt.zero_grad()
                    v1 = augmenter(xb)
                    v2 = augmenter(xb)
                    z1 = self.projection_head(self.encoder.extract_features(v1.to(device)))
                    z2 = self.projection_head(self.encoder.extract_features(v2.to(device)))
                    loss = loss_fn(z1, z2)
                    loss.backward()
                    opt.step()

        elif self.pretext_task in ["masked_temporal", "masked_channel"]:
            channels, seq_len = X_tr_unlabeled.shape[1], X_tr_unlabeled.shape[2]
            self.decoder = nn.Sequential(
                nn.Linear(128 * 16, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, channels * seq_len),
            ).to(device)

            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            opt = torch.optim.Adam(params, lr=self.lr)
            crit = nn.MSELoss()

            for _epoch in range(epochs):
                self.encoder.train()
                for xb, _ in loader:
                    opt.zero_grad()
                    xb_masked = xb.clone()
                    if self.pretext_task == "masked_temporal":
                        # Mask 20% temporal span
                        t_len = xb.shape[2]
                        m_len = int(t_len * 0.20)
                        xb_masked[:, :, :m_len] = 0.0
                    else:
                        # Mask 15% channels
                        ch_len = xb.shape[1]
                        m_ch = int(ch_len * 0.15)
                        xb_masked[:, :m_ch, :] = 0.0

                    feat = self.encoder.extract_features(xb_masked.to(device))
                    recon = self.decoder(feat).view(-1, channels, seq_len)
                    loss = crit(recon, xb.to(device))
                    loss.backward()
                    opt.step()

        return self.encoder
