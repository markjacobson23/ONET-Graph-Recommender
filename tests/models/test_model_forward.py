from __future__ import annotations

from torch_geometric.data import HeteroData

from src.models.gnn.hetero_gat import HeteroGATClassifier
from src.models.gnn.hetero_sage import HeteroSAGEClassifier
from src.models.gnn.hetero_trans import HeteroTransformerClassifier


def test_hetero_sage_forward_shape(tiny_heterodata: HeteroData) -> None:
    model = HeteroSAGEClassifier(
        metadata=tiny_heterodata.metadata(),
        hidden_channels=8,
        out_channels=4,
        num_layers=2,
        dropout=0.0,
    )

    logits = model(tiny_heterodata.x_dict, tiny_heterodata.edge_index_dict)

    assert logits.shape == (3, 4)


def test_hetero_transformer_forward_shape(tiny_heterodata: HeteroData) -> None:
    model = HeteroTransformerClassifier(
        metadata=tiny_heterodata.metadata(),
        hidden_channels=8,
        out_channels=4,
        num_layers=1,
        heads=1,
        dropout=0.0,
        edge_dim=2,
    )

    logits = model(
        tiny_heterodata.x_dict,
        tiny_heterodata.edge_index_dict,
        tiny_heterodata.edge_attr_dict,
    )

    assert logits.shape == (3, 4)


def test_hetero_gat_forward_shape(tiny_heterodata: HeteroData) -> None:
    model = HeteroGATClassifier(
        metadata=tiny_heterodata.metadata(),
        hidden_channels=8,
        out_channels=4,
        num_layers=1,
        heads=1,
        dropout=0.0,
        edge_dim=2,
    )

    logits = model(
        tiny_heterodata.x_dict,
        tiny_heterodata.edge_index_dict,
        tiny_heterodata.edge_attr_dict,
    )

    assert logits.shape == (3, 4)
