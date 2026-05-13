from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.models.baselines.occupation_mlp import OccupationMLPClassifier
from src.models.gnn.hetero_sage import HeteroSAGEClassifier
from src.models.gnn.hetero_trans import HeteroTransformerClassifier


def build_model_and_optimizer(
    model_name: str,
    data: HeteroData,
    num_classes: int,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, bool, bool]:
    """Build a model, optimizer, and graph-awareness flags for an experiment."""

    if model_name == "mlp":
        model = OccupationMLPClassifier(
            in_channels=data["occupation"].num_node_features,
            hidden_channels=64,
            out_channels=num_classes,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )
        return model, optimizer, False, False

    if model_name == "hetero_sage":
        model = HeteroSAGEClassifier(
            metadata=data.metadata(),
            hidden_channels=64,
            out_channels=num_classes,
            num_layers=2,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )
        return model, optimizer, True, False

    if model_name == "hetero_transformer":
        model = HeteroTransformerClassifier(
            metadata=data.metadata(),
            hidden_channels=64,
            out_channels=num_classes,
            num_layers=1,
            heads=1,
            dropout=0.0,
            edge_dim=2,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )
        return model, optimizer, True, True

    raise ValueError(f"Unknown model name: {model_name}")
