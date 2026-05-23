from __future__ import annotations

import torch
from torch import nn


class TemporalRNN(nn.Module):
    """Sequence model for fused pose and motion embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_dim = hidden_dim

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        return output[:, -1, :]
