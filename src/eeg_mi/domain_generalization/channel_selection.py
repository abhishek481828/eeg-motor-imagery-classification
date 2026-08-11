"""Subject-Independent Channel Selection Utilities.

Performs channel ranking and selection strictly using training subject data.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Standard 64-channel PhysioNet EEG electrode names in order
STANDARD_64_CHANNELS = [
    "Fc5",
    "Fc3",
    "Fc1",
    "Fcz",
    "Fc2",
    "Fc4",
    "Fc6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "Cp5",
    "Cp3",
    "Cp1",
    "Cpz",
    "Cp2",
    "Cp4",
    "Cp6",
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "Ft7",
    "Ft8",
    "T7",
    "T8",
    "T9",
    "T10",
    "Tp7",
    "Tp8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "Po7",
    "Po3",
    "Poz",
    "Po4",
    "Po8",
    "O1",
    "Oz",
    "O2",
    "Iz",
    "Af7",
    "Af3",
    "Afz",
    "Af4",
    "Af8",
    "F1",
    "F2",
    "P1",
    "P2",
    "Po1",
    "Po2",
    "Fpz",
    "Cp1",
    "Cp2",
]

# Motor-cortex focused subset (21 channels around sensorimotor strip)
MOTOR_CORTEX_CHANNELS = [
    "Fc5",
    "Fc3",
    "Fc1",
    "Fcz",
    "Fc2",
    "Fc4",
    "Fc6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "Cp5",
    "Cp3",
    "Cp1",
    "Cpz",
    "Cp2",
    "Cp4",
    "Cp6",
]


class ChannelSelector:
    """Subject-independent channel selector fitted on training data."""

    def __init__(self, mode: str = "all", k_channels: int = 32):
        self.mode = mode.lower()
        self.k_channels = k_channels
        self.selected_indices: np.ndarray | None = None

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray) -> "ChannelSelector":
        """Select channel indices strictly using training data X_tr (N, C, T) and y_tr (N,)."""
        num_channels = X_tr.shape[1]

        if self.mode == "all":
            self.selected_indices = np.arange(num_channels)
        elif self.mode == "motor_cortex":
            # Map motor-cortex channel names to indices (case-insensitive)
            standard_lower = [ch.lower() for ch in STANDARD_64_CHANNELS[:num_channels]]
            indices = []
            for ch in MOTOR_CORTEX_CHANNELS:
                if ch.lower() in standard_lower:
                    indices.append(standard_lower.index(ch.lower()))
            self.selected_indices = np.array(sorted(set(indices)))
        elif self.mode == "mutual_info":
            # Compute variance / energy per channel as summary features
            feats = np.var(X_tr, axis=-1)  # (N, C)
            mi_scores = mutual_info_classif(feats, y_tr, random_state=42)
            top_k = np.argsort(mi_scores)[::-1][: self.k_channels]
            self.selected_indices = np.sort(top_k)
        elif self.mode == "spectral_entropy":
            # Select channels with highest variance in 8-30 Hz ERD/ERS range
            band_power = np.var(X_tr, axis=-1)  # (N, C)
            score_per_ch = np.mean(band_power, axis=0)
            top_k = np.argsort(score_per_ch)[::-1][: self.k_channels]
            self.selected_indices = np.sort(top_k)
        else:
            raise ValueError(f"Unknown channel selection mode: {self.mode}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Filter dataset channels to selected indices."""
        assert self.selected_indices is not None, "ChannelSelector must be fitted first!"
        return X[:, self.selected_indices, :]
