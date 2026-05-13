from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from src.analysis.soc_classifier.graph_stats import (
    RESULTS_PATH,
    build_graph_variant_stats,
    write_graph_variant_stats,
)


def _add_edge_pair(
    data: HeteroData,
    src_type: str,
    rel: str,
    dst_type: str,
    edge_index: torch.Tensor,
) -> None:
    edge_attr = torch.tensor(
        [[1.0, 2.0] for _ in range(edge_index.size(1))],
        dtype=torch.float,
    )

    data[src_type, rel, dst_type].edge_index = edge_index
    data[src_type, rel, dst_type].edge_attr = edge_attr
    data[dst_type, f"rev_{rel}", src_type].edge_index = torch.flip(edge_index, dims=[0])
    data[dst_type, f"rev_{rel}", src_type].edge_attr = edge_attr


def _build_graph(forward_edges_per_type: int) -> HeteroData:
    data = HeteroData()
    data["occupation"].x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    data["skill"].x = torch.tensor([[1.0], [2.0]], dtype=torch.float)
    data["knowledge"].x = torch.tensor([[1.0]], dtype=torch.float)
    data["ability"].x = torch.tensor([[1.0]], dtype=torch.float)

    data["occupation"].num_nodes = 3
    data["skill"].num_nodes = 2
    data["knowledge"].num_nodes = 1
    data["ability"].num_nodes = 1

    edge_specs = {
        ("occupation", "requires_skill", "skill"): torch.tensor(
            [[0, 1, 2], [0, 1, 0]], dtype=torch.long
        ),
        ("occupation", "requires_knowledge", "knowledge"): torch.tensor(
            [[0, 1, 2], [0, 0, 0]], dtype=torch.long
        ),
        ("occupation", "requires_ability", "ability"): torch.tensor(
            [[0, 1, 2], [0, 0, 0]], dtype=torch.long
        ),
    }

    for edge_type, edge_index in edge_specs.items():
        _add_edge_pair(
            data,
            edge_type[0],
            edge_type[1],
            edge_type[2],
            edge_index[:, :forward_edges_per_type],
        )

    return data


def _build_variant_graphs() -> dict[str, HeteroData]:
    return {
        "dense": _build_graph(forward_edges_per_type=3),
        "core_broad": _build_graph(forward_edges_per_type=2),
        "core_strict": _build_graph(forward_edges_per_type=1),
    }


def test_build_graph_variant_stats_computes_retention() -> None:
    stats_df = build_graph_variant_stats(_build_variant_graphs())

    assert list(stats_df.columns) == [
        "graph_variant",
        "num_occupation_nodes",
        "num_skill_nodes",
        "num_knowledge_nodes",
        "num_ability_nodes",
        "num_edges_by_type",
        "num_forward_edges",
        "num_reverse_edges",
        "num_total_edges",
        "dense_forward_edges",
        "forward_edges_retained_from_dense",
        "forward_edge_retention_pct",
        "dense_total_edges",
        "total_edges_retained_from_dense",
        "total_edge_retention_pct",
    ]

    dense_row = stats_df[stats_df["graph_variant"] == "dense"].iloc[0]
    assert dense_row["num_occupation_nodes"] == 3
    assert dense_row["num_skill_nodes"] == 2
    assert dense_row["num_knowledge_nodes"] == 1
    assert dense_row["num_ability_nodes"] == 1
    assert dense_row["num_forward_edges"] == 9
    assert dense_row["num_reverse_edges"] == 9
    assert dense_row["num_total_edges"] == 18
    assert dense_row["dense_forward_edges"] == 9
    assert dense_row["forward_edges_retained_from_dense"] == 9
    assert dense_row["forward_edge_retention_pct"] == pytest.approx(1.0)
    assert dense_row["dense_total_edges"] == 18
    assert dense_row["total_edges_retained_from_dense"] == 18
    assert dense_row["total_edge_retention_pct"] == pytest.approx(1.0)
    assert json.loads(dense_row["num_edges_by_type"]) == {
        "occupation__requires_ability__ability": 3,
        "occupation__requires_knowledge__knowledge": 3,
        "occupation__requires_skill__skill": 3,
        "ability__rev_requires_ability__occupation": 3,
        "knowledge__rev_requires_knowledge__occupation": 3,
        "skill__rev_requires_skill__occupation": 3,
    }

    broad_row = stats_df[stats_df["graph_variant"] == "core_broad"].iloc[0]
    assert broad_row["forward_edges_retained_from_dense"] == 6
    assert broad_row["forward_edge_retention_pct"] == pytest.approx(6 / 9)
    assert broad_row["total_edges_retained_from_dense"] == 12
    assert broad_row["total_edge_retention_pct"] == pytest.approx(12 / 18)

    strict_row = stats_df[stats_df["graph_variant"] == "core_strict"].iloc[0]
    assert strict_row["forward_edges_retained_from_dense"] == 3
    assert strict_row["forward_edge_retention_pct"] == pytest.approx(3 / 9)
    assert strict_row["total_edges_retained_from_dense"] == 6
    assert strict_row["total_edge_retention_pct"] == pytest.approx(6 / 18)


def test_write_graph_variant_stats_writes_csv(tmp_path: Path) -> None:
    processed_graphs_dir = tmp_path / "graphs"
    processed_graphs_dir.mkdir(parents=True, exist_ok=True)

    for graph_variant, graph in _build_variant_graphs().items():
        torch.save(graph, processed_graphs_dir / f"{graph_variant}_heterodata.pt")

    output_path = (
        tmp_path / "results" / "soc_classifier" / "graph_variant_stats.csv"
    )
    saved_path = write_graph_variant_stats(
        processed_graphs_dir=processed_graphs_dir,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    stats_df = pd.read_csv(output_path)
    assert list(stats_df["graph_variant"]) == [
        "dense",
        "core_broad",
        "core_strict",
    ]
    assert "num_edges_by_type" in stats_df.columns
    assert RESULTS_PATH == Path("data/processed/results/soc_classifier/graph_variant_stats.csv")
