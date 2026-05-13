from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from torch_geometric.data import HeteroData

from src.core.config import load_config, resolve_project_path
from src.graph.data import load_heterodata


RESULTS_PATH = Path("data/processed/results/soc_classifier/graph_variant_stats.csv")
GRAPH_VARIANTS = ["dense", "core_broad", "core_strict"]
NODE_TYPES = ["occupation", "skill", "knowledge", "ability"]


def _num_nodes(data: HeteroData, node_type: str) -> int:
    """Return the number of nodes for a node type, falling back to x.shape."""

    if node_type not in data.node_types:
        return 0

    node_store = data[node_type]
    if getattr(node_store, "num_nodes", None) is not None:
        return int(node_store.num_nodes)
    if getattr(node_store, "x", None) is not None:
        return int(node_store.x.size(0))
    return 0


def _edge_count_by_type(data: HeteroData) -> dict[str, int]:
    """Return a stable mapping of edge type string to edge count."""

    edge_counts: dict[str, int] = {}
    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_key = "__".join(edge_type)
        edge_counts[edge_key] = int(edge_store.edge_index.size(1))
    return dict(sorted(edge_counts.items()))


def summarize_graph_variant(graph_variant: str, data: HeteroData) -> dict[str, object]:
    """Summarize a single graph variant."""

    edge_counts = _edge_count_by_type(data)
    forward_edge_counts = {
        edge_type: count
        for edge_type, count in edge_counts.items()
        if "__rev_" not in edge_type
    }
    reverse_edge_counts = {
        edge_type: count
        for edge_type, count in edge_counts.items()
        if "__rev_" in edge_type
    }

    num_forward_edges = sum(forward_edge_counts.values())
    num_reverse_edges = sum(reverse_edge_counts.values())
    num_total_edges = num_forward_edges + num_reverse_edges

    return {
        "graph_variant": graph_variant,
        "num_occupation_nodes": _num_nodes(data, "occupation"),
        "num_skill_nodes": _num_nodes(data, "skill"),
        "num_knowledge_nodes": _num_nodes(data, "knowledge"),
        "num_ability_nodes": _num_nodes(data, "ability"),
        "num_edges_by_type": json.dumps(edge_counts, sort_keys=True),
        "num_forward_edges": num_forward_edges,
        "num_reverse_edges": num_reverse_edges,
        "num_total_edges": num_total_edges,
    }


def build_graph_variant_stats(
    graphs_by_variant: dict[str, HeteroData],
) -> pd.DataFrame:
    """Build a table of graph stats and edge-retention counts."""

    if "dense" not in graphs_by_variant:
        raise ValueError("dense graph variant is required for retention counts")

    dense_stats = summarize_graph_variant("dense", graphs_by_variant["dense"])
    dense_forward_edges = int(dense_stats["num_forward_edges"])
    dense_total_edges = int(dense_stats["num_total_edges"])

    rows: list[dict[str, object]] = []
    for graph_variant in GRAPH_VARIANTS:
        data = graphs_by_variant[graph_variant]
        stats = summarize_graph_variant(graph_variant, data)
        forward_edges = int(stats["num_forward_edges"])
        total_edges = int(stats["num_total_edges"])

        stats["dense_forward_edges"] = dense_forward_edges
        stats["forward_edges_retained_from_dense"] = forward_edges
        stats["forward_edge_retention_pct"] = (
            forward_edges / dense_forward_edges if dense_forward_edges > 0 else 0.0
        )
        stats["dense_total_edges"] = dense_total_edges
        stats["total_edges_retained_from_dense"] = total_edges
        stats["total_edge_retention_pct"] = (
            total_edges / dense_total_edges if dense_total_edges > 0 else 0.0
        )
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df = stats_df[
            [
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
        ]
    return stats_df


def load_graph_variants(
    processed_graphs_dir: Path,
) -> dict[str, HeteroData]:
    """Load the saved HeteroData graphs for each graph variant."""

    return {
        graph_variant: load_heterodata(processed_graphs_dir / f"{graph_variant}_heterodata.pt")
        for graph_variant in GRAPH_VARIANTS
    }


def write_graph_variant_stats(
    processed_graphs_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Load graphs, compute stats, and write the CSV."""

    output_path = resolve_project_path(output_path or RESULTS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graphs_by_variant = load_graph_variants(processed_graphs_dir)
    stats_df = build_graph_variant_stats(graphs_by_variant)
    stats_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Generate graph variant stats from the saved heterodata graphs."""

    config = load_config()
    processed_graphs_dir = resolve_project_path(config["paths"]["processed_graphs_dir"])
    output_path = write_graph_variant_stats(processed_graphs_dir=processed_graphs_dir)
    print(f"Wrote graph variant stats to {output_path}")


if __name__ == "__main__":
    main()
