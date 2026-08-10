"""Riemannian Geometry Covariance Feature Extractor (Numpy/Scipy).

Estimates covariance matrices for EEG epochs and projects them into
Euclidean Tangent Space anchored at a training-fitted reference mean matrix.
"""

import numpy as np
import scipy.linalg as la


class RiemannianTangentSpaceTransformer:
    """Leakage-safe Riemannian Covariance Tangent Space Transformer."""

    def __init__(self, reg_eps: float = 1e-5):
        self.reg_eps = reg_eps
        self.C_ref = None
        self.C_ref_inv_sqrt = None

    def fit(self, X: np.ndarray) -> "RiemannianTangentSpaceTransformer":
        """Fit reference mean covariance matrix C_ref strictly on X_train.

        Args:
            X: Training EEG dataset of shape (num_epochs, channels, time)
        """
        covs = [self._estimate_cov(X[i]) for i in range(len(X))]
        
        # Log-Euclidean Reference Mean
        log_covs = [la.logm(c) for c in covs]
        mean_log = np.mean(log_covs, axis=0)
        self.C_ref = la.expm(mean_log)

        # C_ref^(-1/2) for tangent space projection
        vals, vecs = la.eigh(self.C_ref)
        vals = np.maximum(vals, 1e-10)
        self.C_ref_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project dataset covariance matrices into tangent space.

        Args:
            X: EEG dataset of shape (num_epochs, channels, time)
        """
        assert self.C_ref_inv_sqrt is not None, "Transformer must be fitted first!"
        num_epochs = X.shape[0]
        feats = [self._project_epoch(X[i]) for i in range(num_epochs)]
        return np.vstack(feats)

    def _estimate_cov(self, epoch: np.ndarray) -> np.ndarray:
        """Estimate regularized sample covariance matrix."""
        channels, time_points = epoch.shape
        # Zero-mean channels
        epoch_centered = epoch - np.mean(epoch, axis=1, keepdims=True)
        cov = (epoch_centered @ epoch_centered.T) / (time_points - 1)
        # Regularize to ensure positive definiteness
        cov += self.reg_eps * np.eye(channels)
        return cov

    def _project_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """Project single epoch covariance into tangent space vector."""
        cov = self._estimate_cov(epoch)
        # Whitening by C_ref^(-1/2)
        whitened = self.C_ref_inv_sqrt @ cov @ self.C_ref_inv_sqrt
        # Matrix logarithm
        tangent_mat = la.logm(whitened)
        
        # Extract upper triangular vector with off-diagonal sqrt(2) scaling
        channels = tangent_mat.shape[0]
        vec = []
        for i in range(channels):
            vec.append(tangent_mat[i, i])
            for j in range(i + 1, channels):
                vec.append(np.sqrt(2.0) * tangent_mat[i, j])
        return np.array(vec, dtype=np.float32)
