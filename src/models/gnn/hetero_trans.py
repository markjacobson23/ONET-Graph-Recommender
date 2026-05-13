import torch
from torch_geometric.nn import HeteroConv, TransformerConv
import torch.nn.functional as F


class HeteroTransformerClassifier(torch.nn.Module):

    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=2,
        dropout=0.2,
        edge_dim=2,
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: TransformerConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    )
                    for edge_type in metadata[1]
                },
                aggr="sum",
            )

            self.convs.append(conv)

        self.classifier = torch.nn.Linear(
            hidden_channels,
            out_channels,
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
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