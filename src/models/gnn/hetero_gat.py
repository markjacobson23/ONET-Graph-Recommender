from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class HeteroGATClassifier(torch.nn.Module):
    """A heterogeneous GAT classifier that uses edge attributes."""

    def __init__(
        self,
        metadata: tuple[dict, dict],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 2,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        add_self_loops=False,
                    )
                    for edge_type in metadata[1]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict,
    ) -> torch.Tensor:
        """Run the hetero GAT forward pass."""

        for conv in self.convs:
            x_dict = conv(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            x_dict = {
                node_type: F.dropout(
                    F.relu(x),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, x in x_dict.items()
            }

        return self.classifier(x_dict["occupation"])
