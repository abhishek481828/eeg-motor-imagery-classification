"""LSTM baseline model for temporal signal modeling."""

import torch
import torch.nn as nn


class EEGLSTM(nn.Module):
    """Recurrent LSTM Model for temporal EEG sequence modeling."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch, channels, time) -> transpose to (batch, time, channels)
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        # Use final time-step hidden state
        last_out = out[:, -1, :]
        return self.fc(last_out)
