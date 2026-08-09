"""Hybrid CNN-LSTM Deep Learning Architecture for EEG Motor Imagery."""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM Deep Neural Network.

    Extracts spatial-temporal features using 1D convolutional layers,
    followed by LSTM layers to capture temporal sequence dependencies.
    """

    def __init__(
        self,
        in_channels: int = 64,
        sequence_length: int = 480,
        num_classes: int = 4,
        cnn_filters: list[int] | None = None,
        kernel_size: int = 15,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [32, 64, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes

        # CNN Spatial-Temporal Feature Extractor
        conv_layers = []
        c_in = in_channels
        for c_out in cnn_filters:
            conv_layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout / 2),
                ]
            )
            c_in = c_out
        self.cnn = nn.Sequential(*conv_layers)

        # LSTM Sequence Processor
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects input tensor of shape (batch, channels, time)."""
        # CNN feature extraction: (batch, channels, time) -> (batch, cnn_out, time_pooled)
        feat = self.cnn(x)
        # Transpose for LSTM: (batch, time_pooled, cnn_out)
        feat = feat.transpose(1, 2)
        # LSTM sequence processing
        lstm_out, _ = self.lstm(feat)
        # Global temporal pooling or last timestep
        out = lstm_out[:, -1, :]
        # Classifier output
        logits = self.classifier(out)
        return logits
