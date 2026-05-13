import torch.nn as nn
import torch.nn.functional as F


class OccupationMLPClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.2,
    ):
        super().__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x
