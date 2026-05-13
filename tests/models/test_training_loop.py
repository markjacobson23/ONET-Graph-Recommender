from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.evaluation.build_model import build_model_and_optimizer
from src.models.evaluation.train_model import (
    compute_classification_metrics,
    evaluate,
    get_model_logits,
)


class RecordingModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.calls: list[tuple[object, ...]] = []

    def forward(self, *args):  # type: ignore[override]
        self.calls.append(args)
        return self.logits


def _build_classification_data() -> HeteroData:
    data = HeteroData()
    data["occupation"].x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    data["occupation"].y = torch.tensor([0, 1, 1], dtype=torch.long)
    data["occupation"].train_mask = torch.tensor([True, True, False])
    data["occupation"].val_mask = torch.tensor([False, False, True])
    data["occupation"].test_mask = torch.tensor([False, False, True])
    data["occupation"].num_nodes = 3
    return data


def test_get_model_logits_dispatches_correctly(tiny_heterodata: HeteroData) -> None:
    logits = torch.zeros((3, 2), dtype=torch.float)

    mlp_model = RecordingModel(logits)
    result = get_model_logits(
        mlp_model,
        tiny_heterodata,
        graph_aware=False,
        edge_aware=False,
    )
    assert result is logits
    assert len(mlp_model.calls) == 1
    assert torch.equal(mlp_model.calls[0][0], tiny_heterodata["occupation"].x)

    sage_model = RecordingModel(logits)
    result = get_model_logits(
        sage_model,
        tiny_heterodata,
        graph_aware=True,
        edge_aware=False,
    )
    assert result is logits
    assert len(sage_model.calls) == 1
    assert sage_model.calls[0][0].keys() == tiny_heterodata.x_dict.keys()
    assert sage_model.calls[0][1].keys() == tiny_heterodata.edge_index_dict.keys()
    for key in tiny_heterodata.x_dict:
        assert torch.equal(sage_model.calls[0][0][key], tiny_heterodata.x_dict[key])
    for key in tiny_heterodata.edge_index_dict:
        assert torch.equal(
            sage_model.calls[0][1][key],
            tiny_heterodata.edge_index_dict[key],
        )

    trans_model = RecordingModel(logits)
    result = get_model_logits(
        trans_model,
        tiny_heterodata,
        graph_aware=True,
        edge_aware=True,
    )
    assert result is logits
    assert len(trans_model.calls) == 1
    assert trans_model.calls[0][0].keys() == tiny_heterodata.x_dict.keys()
    assert trans_model.calls[0][1].keys() == tiny_heterodata.edge_index_dict.keys()
    assert trans_model.calls[0][2].keys() == tiny_heterodata.edge_attr_dict.keys()
    for key in tiny_heterodata.x_dict:
        assert torch.equal(trans_model.calls[0][0][key], tiny_heterodata.x_dict[key])
    for key in tiny_heterodata.edge_index_dict:
        assert torch.equal(
            trans_model.calls[0][1][key],
            tiny_heterodata.edge_index_dict[key],
        )
    for key in tiny_heterodata.edge_attr_dict:
        assert torch.equal(
            trans_model.calls[0][2][key],
            tiny_heterodata.edge_attr_dict[key],
        )

    with pytest.raises(ValueError, match="edge_aware=True requires graph_aware=True"):
        get_model_logits(
            RecordingModel(logits),
            tiny_heterodata,
            graph_aware=False,
            edge_aware=True,
        )


def test_compute_classification_metrics_handles_missing_classes() -> None:
    y_true = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    y_pred = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        include_balanced_accuracy=True,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(5.0 / 9.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_evaluate_returns_split_metrics_without_balanced_accuracy() -> None:
    data = _build_classification_data()
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [1.0, 4.0],
            [1.0, 4.0],
        ],
        dtype=torch.float,
    )
    model = RecordingModel(logits)

    results = evaluate(
        model=model,
        data=data,
        graph_aware=False,
        edge_aware=False,
    )

    assert results == {
        "train_accuracy": 1.0,
        "train_macro_f1": 1.0,
        "val_accuracy": 1.0,
        "val_macro_f1": 1.0,
        "test_accuracy": 1.0,
        "test_macro_f1": 1.0,
    }


def test_evaluate_returns_split_metrics_with_balanced_accuracy() -> None:
    data = _build_classification_data()
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [1.0, 4.0],
            [1.0, 4.0],
        ],
        dtype=torch.float,
    )
    model = RecordingModel(logits)

    results = evaluate(
        model=model,
        data=data,
        graph_aware=False,
        edge_aware=False,
        include_balanced_accuracy=True,
    )

    assert results == {
        "train_accuracy": 1.0,
        "train_macro_f1": 1.0,
        "train_balanced_accuracy": 1.0,
        "val_accuracy": 1.0,
        "val_macro_f1": 1.0,
        "val_balanced_accuracy": 1.0,
        "test_accuracy": 1.0,
        "test_macro_f1": 1.0,
        "test_balanced_accuracy": 1.0,
    }


def test_build_model_and_optimizer_returns_expected_flags(tiny_heterodata: HeteroData) -> None:
    model, optimizer, graph_aware, edge_aware = build_model_and_optimizer(
        "mlp",
        tiny_heterodata,
        num_classes=2,
    )
    assert graph_aware is False
    assert edge_aware is False
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)
