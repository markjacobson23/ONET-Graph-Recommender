from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.evaluation import compare_baselines_vs_gnn as comparison


def test_run_seed_returns_flat_rows(monkeypatch, tiny_heterodata) -> None:
    data = tiny_heterodata

    def fake_load_heterodata(graph_path: Path):
        return data

    def fake_load_occupation_nodes(featured_nodes_dir: Path):
        return pd.DataFrame(
            {
                "occupation_idx": [0, 1, 2],
                "onetsoc_code": ["15-2051.00", "17-2141.02", "11-0000.00"],
                "occupation_title": [
                    "Data Scientist",
                    "Engineer",
                    "Manager",
                ],
            }
        )

    def fake_add_occupation_labels(data, occupation_nodes):
        data["occupation"].y = torch.tensor([0, 1, 1], dtype=torch.long)
        return data, {"group_a": 0, "group_b": 1}, {0: "group_a", 1: "group_b"}

    def fake_add_train_val_test_masks(data, train_ratio, val_ratio, seed):
        return data

    def fake_majority_class_baseline(data, include_balanced_accuracy=False):
        assert include_balanced_accuracy is True
        return {
            "train_accuracy": 0.5,
            "val_accuracy": 0.25,
            "test_accuracy": 0.75,
            "train_macro_f1": 0.4,
            "val_macro_f1": 0.2,
            "test_macro_f1": 0.6,
            "train_balanced_accuracy": 0.5,
            "val_balanced_accuracy": 0.25,
            "test_balanced_accuracy": 0.75,
        }

    def fake_build_model_and_optimizer(model_name, data, num_classes):
        model = torch.nn.Linear(1, num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return model, optimizer, False, False

    def fake_train_with_early_stopping(
        model,
        data,
        optimizer,
        graph_aware,
        edge_aware,
        include_balanced_accuracy=False,
        num_epochs=2500,
        patience=200,
        print_every=100,
    ):
        assert include_balanced_accuracy is True
        return {
            "results": {
                "train_accuracy": 0.9,
                "val_accuracy": 0.8,
                "test_accuracy": 0.7,
                "train_macro_f1": 0.88,
                "val_macro_f1": 0.77,
                "test_macro_f1": 0.66,
                "train_balanced_accuracy": 0.91,
                "val_balanced_accuracy": 0.81,
                "test_balanced_accuracy": 0.71,
                "best_epoch": 12,
                "best_val_accuracy": 0.8,
            }
        }

    monkeypatch.setattr(comparison, "load_heterodata", fake_load_heterodata)
    monkeypatch.setattr(
        comparison, "load_occupation_nodes", fake_load_occupation_nodes
    )
    monkeypatch.setattr(
        comparison, "add_occupation_labels", fake_add_occupation_labels
    )
    monkeypatch.setattr(
        comparison, "add_train_val_test_masks", fake_add_train_val_test_masks
    )
    monkeypatch.setattr(
        comparison,
        "majority_class_baseline",
        fake_majority_class_baseline,
    )
    monkeypatch.setattr(
        comparison,
        "build_model_and_optimizer",
        fake_build_model_and_optimizer,
    )
    monkeypatch.setattr(
        comparison,
        "train_with_early_stopping",
        fake_train_with_early_stopping,
    )

    rows = comparison.run_seed(
        seed=7,
        graph_variant="dense",
        graph_path=Path("/tmp/dense.pt"),
        featured_nodes_dir=Path("/tmp/featured"),
        model_names=["mlp"],
        include_balanced_accuracy=True,
    )

    assert rows == [
        {
            "graph_variant": "dense",
            "model_name": "majority_class",
            "seed": 7,
            "train_accuracy": 0.5,
            "train_macro_f1": 0.4,
            "train_balanced_accuracy": 0.5,
            "val_accuracy": 0.25,
            "val_macro_f1": 0.2,
            "val_balanced_accuracy": 0.25,
            "test_accuracy": 0.75,
            "test_macro_f1": 0.6,
            "test_balanced_accuracy": 0.75,
            "best_epoch": None,
            "best_val_accuracy": None,
        },
        {
            "graph_variant": "dense",
            "model_name": "mlp",
            "seed": 7,
            "train_accuracy": 0.9,
            "train_macro_f1": 0.88,
            "train_balanced_accuracy": 0.91,
            "val_accuracy": 0.8,
            "val_macro_f1": 0.77,
            "val_balanced_accuracy": 0.81,
            "test_accuracy": 0.7,
            "test_macro_f1": 0.66,
            "test_balanced_accuracy": 0.71,
            "best_epoch": 12,
            "best_val_accuracy": 0.8,
        },
    ]


def test_save_results_csv_writes_expected_columns(tmp_path) -> None:
    rows = [
        {
            "graph_variant": "dense",
            "model_name": "majority_class",
            "seed": 0,
            "train_accuracy": 0.5,
            "train_macro_f1": 0.4,
            "train_balanced_accuracy": 0.5,
            "val_accuracy": 0.25,
            "val_macro_f1": 0.2,
            "val_balanced_accuracy": 0.25,
            "test_accuracy": 0.75,
            "test_macro_f1": 0.6,
            "test_balanced_accuracy": 0.75,
            "best_epoch": None,
            "best_val_accuracy": None,
        }
    ]

    output_path = (
        tmp_path / "soc_classifier" / "soc_classifier_multiseed_results.csv"
    )
    comparison.save_results_csv(
        rows=rows,
        output_path=output_path,
        include_balanced_accuracy=True,
    )

    assert output_path.exists()

    results_df = pd.read_csv(output_path)
    assert list(results_df.columns) == comparison.result_columns(
        include_balanced_accuracy=True
    )
    row = results_df.iloc[0]
    assert row["graph_variant"] == "dense"
    assert row["model_name"] == "majority_class"
    assert row["seed"] == 0
    assert row["train_accuracy"] == pytest.approx(0.5)
    assert row["train_macro_f1"] == pytest.approx(0.4)
    assert row["train_balanced_accuracy"] == pytest.approx(0.5)
    assert row["val_accuracy"] == pytest.approx(0.25)
    assert row["val_macro_f1"] == pytest.approx(0.2)
    assert row["val_balanced_accuracy"] == pytest.approx(0.25)
    assert row["test_accuracy"] == pytest.approx(0.75)
    assert row["test_macro_f1"] == pytest.approx(0.6)
    assert row["test_balanced_accuracy"] == pytest.approx(0.75)
    assert pd.isna(row["best_epoch"])
    assert pd.isna(row["best_val_accuracy"])
