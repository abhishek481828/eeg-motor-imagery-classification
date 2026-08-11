"""Domain Generalization Loss Modules (CORAL & MMD).

Computes domain-invariance objectives between training subject distributions
to encourage domain-agnostic feature representations.
"""

import torch
import torch.nn as nn


class CORALLoss(nn.Module):
    """Correlation Alignment (CORAL) Loss.

    Aligns second-order statistics (covariance matrices) of feature representations
    between pair-wise training domains.
    """

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute CORAL loss between source and target feature batches.

        Args:
            source_features: Tensor of shape (N_s, D)
            target_features: Tensor of shape (N_t, D)
        """
        d = source_features.size(1)
        if d == 0 or source_features.size(0) <= 1 or target_features.size(0) <= 1:
            return torch.tensor(0.0, device=source_features.device)

        # Covariance matrices
        cov_s = self._compute_cov(source_features)
        cov_t = self._compute_cov(target_features)

        # Frobenius norm distance
        loss = torch.norm(cov_s - cov_t, p="fro") ** 2
        loss = loss / (4.0 * d * d)
        return loss

    def _compute_cov(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x_centered = x - torch.mean(x, dim=0, keepdim=True)
        cov = torch.matmul(x_centered.t(), x_centered) / (n - 1.0)
        return cov


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) Loss with RBF Gaussian Kernels."""

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(
        self, source: torch.Tensor, target: torch.Tensor, fix_sigma: float | None = None
    ) -> torch.Tensor:
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        total_outer = torch.matmul(total, total.t())
        total_sq = torch.sum(total**2, dim=1, keepdim=True)
        distance_matrix = total_sq + total_sq.t() - 2.0 * total_outer

        bandwidth: torch.Tensor | float
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(distance_matrix.detach()) / (n_samples**2 - n_samples + 1e-8)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-distance_matrix / bw) for bw in bandwidth_list]
        res: torch.Tensor = torch.stack(kernel_val).sum(dim=0)
        return res

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute MMD loss between source and target feature batches."""
        if source_features.size(0) <= 1 or target_features.size(0) <= 1:
            return torch.tensor(0.0, device=source_features.device)

        n_s = source_features.size(0)

        kernels = self.gaussian_kernel(source_features, target_features)
        XX = kernels[:n_s, :n_s]
        YY = kernels[n_s:, n_s:]
        XY = kernels[:n_s, n_s:]

        loss = torch.mean(XX) + torch.mean(YY) - 2.0 * torch.mean(XY)
        return torch.clamp(loss, min=0.0)
