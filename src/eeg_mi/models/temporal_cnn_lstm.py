"""True Temporal Sequence CNN-LSTM Architecture.

Splits input EEG epoch into a temporal sequence of sliding sub-windows,
extracts spatial-temporal features per sub-window using a 1D-CNN backbone,
and processes the sequence with an LSTM layer to model temporal transitions.
"""

import torch
import torch.nn as nn


class TemporalCNNLSTM(nn.Module):
    """Sequence-preserving CNN-LSTM for multi-channel EEG signals."""

    def __init__(
        self,
        in_channels: int = 64,
        sequence_length: int = 481,
        num_sub_windows: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_sub_windows = num_sub_windows
        self.hidden_dim = hidden_dim

        # 1D-CNN Sub-window Feature Extractor
        self.sub_cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.feature_dim = 64 * 8

        # LSTM Temporal Model
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input EEG tensor of shape (batch, channels, time)
        """
        batch, channels, time_steps = x.shape
        sub_len = time_steps // self.num_sub_windows

        # Extract features per sub-window sequence
        seq_features = []
        for i in range(self.num_sub_windows):
            start = i * sub_len
            end = start + sub_len if i < self.num_sub_windows - 1 else time_steps
            sub_x = x[:, :, start:end]
            f_sub = self.sub_cnn(sub_x).view(batch, -1)
            seq_features.append(f_sub)

        # Sequence tensor of shape (batch, num_sub_windows, feature_dim)
        seq_tensor = torch.stack(seq_features, dim=1)

        # LSTM forward pass
        lstm_out, (h_n, _) = self.lstm(seq_tensor)
        # Use final hidden state
        last_hidden = h_n[-1]

        return self.classifier(last_hidden)
