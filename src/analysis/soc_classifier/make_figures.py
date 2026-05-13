from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from src.core.config import resolve_project_path


RESULTS_DIR = Path("data/processed/results/soc_classifier")
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "soc_classifier_multiseed_summary.csv"
GRAPH_STATS_PATH = RESULTS_DIR / "graph_variant_stats.csv"

TEST_ACCURACY_FILENAME = "test_accuracy_by_model.png"
MACRO_F1_FILENAME = "macro_f1_by_model.png"
EDGE_COUNTS_FILENAME = "graph_variant_edge_counts.png"
PIPELINE_FLOW_FILENAME = "pipeline_flow.png"
GRAPH_SCHEMA_FILENAME = "graph_schema.png"

DISPLAY_MODELS = [
    {
        "display_label": "Majority class",
        "graph_variant": "dense",
        "model_name": "majority_class",
    },
    {
        "display_label": "MLP",
        "graph_variant": "dense",
        "model_name": "mlp",
    },
    {
        "display_label": "Dense HeteroSAGE",
        "graph_variant": "dense",
        "model_name": "hetero_sage",
    },
    {
        "display_label": "Core strict HeteroSAGE",
        "graph_variant": "core_strict",
        "model_name": "hetero_sage",
    },
    {
        "display_label": "Dense HeteroTransformer",
        "graph_variant": "dense",
        "model_name": "hetero_transformer",
    },
]

MODEL_COLORS = {
    "Majority class": "#6b7280",
    "MLP": "#2563eb",
    "Dense HeteroSAGE": "#16a34a",
    "Core strict HeteroSAGE": "#ea580c",
    "Dense HeteroTransformer": "#7c3aed",
}


def load_summary_table(summary_path: Path) -> pd.DataFrame:
    """Load the aggregated model summary CSV."""

    return pd.read_csv(summary_path)


def load_graph_stats(graph_stats_path: Path) -> pd.DataFrame:
    """Load the graph variant stats CSV."""

    return pd.read_csv(graph_stats_path)


def prepare_model_metric_table(
    summary_df: pd.DataFrame,
    metric_prefix: str,
) -> pd.DataFrame:
    """Pick the rows that should appear in the model comparison charts."""

    rows: list[dict[str, object]] = []
    for selection in DISPLAY_MODELS:
        matching_rows = summary_df[
            (summary_df["graph_variant"] == selection["graph_variant"])
            & (summary_df["model_name"] == selection["model_name"])
        ]
        if matching_rows.empty:
            raise ValueError(
                "Missing summary row for "
                f"{selection['graph_variant']} / {selection['model_name']}"
            )

        row = matching_rows.iloc[0]
        rows.append(
            {
                "display_label": selection["display_label"],
                "graph_variant": selection["graph_variant"],
                "model_name": selection["model_name"],
                "mean": float(row[f"{metric_prefix}_mean"]),
                "std": float(row[f"{metric_prefix}_std"]),
            }
        )

    return pd.DataFrame(rows)


def _style_metric_axis(ax: plt.Axes, xlabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_metric_chart(
    metric_table: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle: str,
    xlabel: str,
) -> None:
    """Render a horizontal bar chart with error bars."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metric_table["display_label"].tolist()
    means = metric_table["mean"].astype(float).mul(100.0).tolist()
    stds = metric_table["std"].astype(float).mul(100.0).tolist()
    colors = [MODEL_COLORS.get(label, "#3b82f6") for label in labels]
    y_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        y_positions,
        means,
        xerr=stds,
        color=colors,
        edgecolor="#111827",
        capsize=6,
        alpha=0.95,
    )
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_title(f"{title}\n{subtitle}", pad=18, fontsize=15, weight="bold")
    _style_metric_axis(ax, xlabel)

    max_value = max(m + s for m, s in zip(means, stds, strict=True)) if means else 0.0
    ax.set_xlim(0, max(50.0, max_value * 1.18))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_test_accuracy_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create the test accuracy comparison chart."""

    metric_table = prepare_model_metric_table(summary_df, "test_accuracy")
    plot_model_metric_chart(
        metric_table=metric_table,
        output_path=output_path,
        title="SOC Classifier Test Accuracy by Model / Graph Variant",
        subtitle="Mean ± std across five random splits",
        xlabel="Test accuracy (%)",
    )


def build_macro_f1_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create the macro F1 comparison chart."""

    metric_table = prepare_model_metric_table(summary_df, "test_macro_f1")
    plot_model_metric_chart(
        metric_table=metric_table,
        output_path=output_path,
        title="SOC Classifier Macro F1 by Model / Graph Variant",
        subtitle="Mean ± std across five random splits",
        xlabel="Macro F1 (%)",
    )


def prepare_edge_count_table(graph_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a chart-ready graph stats table."""

    ordered_rows: list[dict[str, object]] = []
    for graph_variant in ["dense", "core_broad", "core_strict"]:
        matching_rows = graph_stats_df[graph_stats_df["graph_variant"] == graph_variant]
        if matching_rows.empty:
            raise ValueError(f"Missing graph stats row for {graph_variant}")

        row = matching_rows.iloc[0]
        ordered_rows.append(
            {
                "graph_variant": graph_variant,
                "forward_edges": int(row["num_forward_edges"]),
                "reverse_edges": int(row["num_reverse_edges"]),
                "dense_forward_edges": int(row["dense_forward_edges"]),
                "forward_edges_retained_from_dense": int(
                    row["forward_edges_retained_from_dense"]
                ),
                "total_edges_retained_from_dense": int(
                    row["total_edges_retained_from_dense"]
                ),
                "forward_edge_retention_pct": float(row["forward_edge_retention_pct"]),
                "total_edge_retention_pct": float(row["total_edge_retention_pct"]),
            }
        )

    return pd.DataFrame(ordered_rows)


def build_graph_variant_edge_counts_chart(
    graph_stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the graph-variant forward-edge retention chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_df = prepare_edge_count_table(graph_stats_df)
    labels = chart_df["graph_variant"].str.replace("_", " ", regex=False).tolist()
    forward_retention_pct = chart_df["forward_edge_retention_pct"].tolist()

    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        x_positions,
        [pct * 100.0 for pct in forward_retention_pct],
        color=["#2563eb", "#16a34a", "#ea580c"],
        edgecolor="#111827",
        width=0.6,
    )
    ax.bar_label(
        bars,
        labels=[f"{pct * 100:.1f}%" for pct in forward_retention_pct],
        padding=3,
        fontsize=10,
    )

    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Forward edges retained (%)")
    ax.set_title(
        "Graph Variant Forward-Edge Retention",
        pad=16,
        fontsize=15,
        weight="bold",
    )
    ax.set_ylim(0, max(100.0, max(pct * 100.0 for pct in forward_retention_pct) * 1.22))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#111827",
    fontsize: int = 12,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#111827",
        wrap=True,
    )


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2, color="#374151"),
    )
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.03, text, ha="center", va="bottom", fontsize=10, color="#374151")


def build_pipeline_flow_diagram(output_path: Path) -> None:
    """Create the SOC classifier pipeline flow diagram."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "SOC Classifier: Data Flow",
        fontsize=18,
        weight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.88,
        "Generated graph artifacts feed baseline and GNN benchmarks.",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    flow_boxes = [
        (0.02, 0.43, 0.13, 0.22, "Raw O*NET\nSQLite", "#e0f2fe"),
        (0.18, 0.43, 0.15, 0.22, "Config-driven\nETL", "#dbeafe"),
        (0.36, 0.43, 0.15, 0.22, "Graph-ready\ntables", "#dcfce7"),
        (0.54, 0.43, 0.15, 0.22, "PyG\nHeteroData", "#fef3c7"),
        (0.72, 0.43, 0.15, 0.22, "Models\nMLP / SAGE /\nTransformer", "#fae8ff"),
        (0.90, 0.43, 0.09, 0.22, "SOC\nprediction", "#fee2e2"),
    ]

    for x, y, width, height, text, color in flow_boxes:
        box_fontsize = 10 if text == "SOC\nprediction" else 12
        _draw_box(ax, (x, y), width, height, text, color, fontsize=box_fontsize)

    for left, right in zip(flow_boxes[:-1], flow_boxes[1:], strict=True):
        _draw_arrow(
            ax,
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_graph_schema_diagram(output_path: Path) -> None:
    """Create the SOC classifier graph schema diagram."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title("SOC Classifier: Graph Schema", fontsize=18, weight="bold", pad=18)

    _draw_box(ax, (0.06, 0.40), 0.18, 0.20, "occupation", "#bfdbfe", fontsize=13)
    _draw_box(ax, (0.38, 0.66), 0.18, 0.16, "skill", "#d9f99d", fontsize=12)
    _draw_box(ax, (0.38, 0.42), 0.20, 0.16, "knowledge", "#fde68a", fontsize=12)
    _draw_box(ax, (0.38, 0.18), 0.18, 0.16, "ability", "#fecaca", fontsize=12)

    _draw_arrow(ax, (0.24, 0.50), (0.38, 0.74))
    _draw_arrow(ax, (0.24, 0.50), (0.38, 0.50))
    _draw_arrow(ax, (0.24, 0.50), (0.38, 0.26))

    legend_text = "Edges:\noccupation → skill\noccupation → knowledge\noccupation → ability"
    ax.text(
        0.74,
        0.68,
        legend_text,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f3f4f6", edgecolor="#6b7280"),
    )

    ax.text(
        0.74,
        0.34,
        "edge_attr = [importance, level]",
        fontsize=12,
        weight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#9ca3af"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_all_figures(
    summary_path: Path,
    graph_stats_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate all SOC classifier figures from the analysis CSVs."""

    output_dir = resolve_project_path(output_dir or FIGURES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_table(resolve_project_path(summary_path))
    graph_stats_df = load_graph_stats(resolve_project_path(graph_stats_path))

    outputs = {
        "test_accuracy": output_dir / TEST_ACCURACY_FILENAME,
        "macro_f1": output_dir / MACRO_F1_FILENAME,
        "edge_counts": output_dir / EDGE_COUNTS_FILENAME,
        "pipeline_flow": output_dir / PIPELINE_FLOW_FILENAME,
        "graph_schema": output_dir / GRAPH_SCHEMA_FILENAME,
    }

    build_test_accuracy_chart(summary_df, outputs["test_accuracy"])
    build_macro_f1_chart(summary_df, outputs["macro_f1"])
    build_graph_variant_edge_counts_chart(graph_stats_df, outputs["edge_counts"])
    build_pipeline_flow_diagram(outputs["pipeline_flow"])
    build_graph_schema_diagram(outputs["graph_schema"])

    return outputs


def main() -> None:
    """Generate the SOC classifier analysis figures."""

    outputs = write_all_figures(
        summary_path=RESULTS_DIR / SUMMARY_PATH.name,
        graph_stats_path=RESULTS_DIR / GRAPH_STATS_PATH.name,
    )
    print(f"Wrote figures to {outputs['test_accuracy'].parent}")


if __name__ == "__main__":
    main()
