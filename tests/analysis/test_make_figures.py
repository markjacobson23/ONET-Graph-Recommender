from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.soc_classifier.make_figures import (
    EDGE_COUNTS_FILENAME,
    MACRO_F1_FILENAME,
    GRAPH_SCHEMA_FILENAME,
    PIPELINE_FLOW_FILENAME,
    SUMMARY_PATH,
    TEST_ACCURACY_FILENAME,
    build_graph_schema_diagram,
    build_graph_variant_edge_counts_chart,
    build_macro_f1_chart,
    build_pipeline_flow_diagram,
    build_test_accuracy_chart,
    prepare_edge_count_table,
    prepare_model_metric_table,
    write_all_figures,
)


def _build_summary_df() -> pd.DataFrame:
    rows = [
        {
            "graph_variant": "dense",
            "model_name": "majority_class",
            "test_accuracy_mean": 0.12,
            "test_accuracy_std": 0.01,
            "test_macro_f1_mean": 0.11,
            "test_macro_f1_std": 0.01,
        },
        {
            "graph_variant": "dense",
            "model_name": "mlp",
            "test_accuracy_mean": 0.44,
            "test_accuracy_std": 0.02,
            "test_macro_f1_mean": 0.42,
            "test_macro_f1_std": 0.02,
        },
        {
            "graph_variant": "dense",
            "model_name": "hetero_sage",
            "test_accuracy_mean": 0.25,
            "test_accuracy_std": 0.03,
            "test_macro_f1_mean": 0.24,
            "test_macro_f1_std": 0.03,
        },
        {
            "graph_variant": "core_strict",
            "model_name": "hetero_sage",
            "test_accuracy_mean": 0.39,
            "test_accuracy_std": 0.04,
            "test_macro_f1_mean": 0.36,
            "test_macro_f1_std": 0.04,
        },
        {
            "graph_variant": "dense",
            "model_name": "hetero_transformer",
            "test_accuracy_mean": 0.51,
            "test_accuracy_std": 0.05,
            "test_macro_f1_mean": 0.49,
            "test_macro_f1_std": 0.05,
        },
    ]
    return pd.DataFrame(rows)


def _build_graph_stats_df() -> pd.DataFrame:
    rows = [
        {
            "graph_variant": "core_strict",
            "num_forward_edges": 30,
            "num_reverse_edges": 30,
            "dense_forward_edges": 90,
            "forward_edges_retained_from_dense": 30,
            "total_edges_retained_from_dense": 60,
            "forward_edge_retention_pct": 30 / 90,
            "total_edge_retention_pct": 60 / 180,
        },
        {
            "graph_variant": "dense",
            "num_forward_edges": 90,
            "num_reverse_edges": 90,
            "dense_forward_edges": 90,
            "forward_edges_retained_from_dense": 90,
            "total_edges_retained_from_dense": 180,
            "forward_edge_retention_pct": 1.0,
            "total_edge_retention_pct": 1.0,
        },
        {
            "graph_variant": "core_broad",
            "num_forward_edges": 60,
            "num_reverse_edges": 60,
            "dense_forward_edges": 90,
            "forward_edges_retained_from_dense": 60,
            "total_edges_retained_from_dense": 120,
            "forward_edge_retention_pct": 60 / 90,
            "total_edge_retention_pct": 120 / 180,
        },
    ]
    return pd.DataFrame(rows)


def test_prepare_model_metric_table_orders_selected_models() -> None:
    summary_df = _build_summary_df()

    metric_table = prepare_model_metric_table(summary_df, "test_accuracy")

    assert list(metric_table["display_label"]) == [
        "Majority class",
        "MLP",
        "Dense HeteroSAGE",
        "Core strict HeteroSAGE",
        "Dense HeteroTransformer",
    ]
    assert metric_table.iloc[1]["mean"] == pytest.approx(0.44)
    assert metric_table.iloc[3]["std"] == pytest.approx(0.04)


def test_prepare_edge_count_table_orders_variants() -> None:
    graph_stats_df = _build_graph_stats_df()

    edge_table = prepare_edge_count_table(graph_stats_df)

    assert list(edge_table["graph_variant"]) == [
        "dense",
        "core_broad",
        "core_strict",
    ]
    assert edge_table.iloc[1]["forward_edges_retained_from_dense"] == 60
    assert edge_table.iloc[2]["total_edge_retention_pct"] == pytest.approx(60 / 180)


def test_write_all_figures_creates_expected_pngs(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "soc_classifier"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _build_summary_df()
    graph_stats_df = _build_graph_stats_df()
    summary_path = results_dir / SUMMARY_PATH.name
    graph_stats_path = results_dir / "graph_variant_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    graph_stats_df.to_csv(graph_stats_path, index=False)

    output_dir = results_dir / "figures"
    outputs = write_all_figures(
        summary_path=summary_path,
        graph_stats_path=graph_stats_path,
        output_dir=output_dir,
    )

    assert output_dir.exists()
    assert outputs["test_accuracy"] == output_dir / TEST_ACCURACY_FILENAME
    assert outputs["macro_f1"] == output_dir / MACRO_F1_FILENAME
    assert outputs["edge_counts"] == output_dir / EDGE_COUNTS_FILENAME
    assert outputs["pipeline_flow"] == output_dir / PIPELINE_FLOW_FILENAME
    assert outputs["graph_schema"] == output_dir / GRAPH_SCHEMA_FILENAME

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_build_individual_figures(tmp_path: Path) -> None:
    summary_df = _build_summary_df()
    graph_stats_df = _build_graph_stats_df()

    test_accuracy_path = tmp_path / TEST_ACCURACY_FILENAME
    macro_f1_path = tmp_path / MACRO_F1_FILENAME
    edge_counts_path = tmp_path / EDGE_COUNTS_FILENAME
    pipeline_flow_path = tmp_path / PIPELINE_FLOW_FILENAME
    graph_schema_path = tmp_path / GRAPH_SCHEMA_FILENAME

    build_test_accuracy_chart(summary_df, test_accuracy_path)
    build_macro_f1_chart(summary_df, macro_f1_path)
    build_graph_variant_edge_counts_chart(graph_stats_df, edge_counts_path)
    build_pipeline_flow_diagram(pipeline_flow_path)
    build_graph_schema_diagram(graph_schema_path)

    for path in [
        test_accuracy_path,
        macro_f1_path,
        edge_counts_path,
        pipeline_flow_path,
        graph_schema_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0
