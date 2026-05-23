from __future__ import annotations

import torch
from torch import nn

from src.models.spatial_cnn import SpatialCNN
from src.models.temporal_rnn import TemporalRNN


class HARNet(nn.Module):
    """Hybrid optical-flow CNN plus pose-landmark recurrent classifier."""

    def __init__(
        self,
        num_classes: int = 4,
        pose_dim: int = 33 * 4,
        flow_channels: int = 2,
        cnn_embedding_dim: int = 128,
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        self.motion_encoder = SpatialCNN(flow_channels, cnn_embedding_dim)
        self.temporal = TemporalRNN(
            input_dim=cnn_embedding_dim + pose_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_dim, num_classes),
        )

    def forward(self, flow_sequence: torch.Tensor, pose_sequence: torch.Tensor) -> torch.Tensor:
        """Classify a batch.

        Args:
            flow_sequence: Tensor shaped [batch, time, 2, height, width].
            pose_sequence: Tensor shaped [batch, time, 132].
        """
        batch_size, steps, channels, height, width = flow_sequence.shape
        flat_flow = flow_sequence.reshape(batch_size * steps, channels, height, width)
        motion_features = self.motion_encoder(flat_flow)
        motion_features = motion_features.reshape(batch_size, steps, -1)
        fused = torch.cat([motion_features, pose_sequence], dim=-1)
        temporal_features = self.temporal(fused)
        return self.classifier(temporal_features)

    @torch.no_grad()
    def predict_proba(self, flow_sequence: torch.Tensor, pose_sequence: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self(flow_sequence, pose_sequence)
        return torch.softmax(logits, dim=-1)
