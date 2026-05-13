from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupationMLPClassifier(nn.Module):
    """A small two-layer MLP for occupation classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass."""

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
