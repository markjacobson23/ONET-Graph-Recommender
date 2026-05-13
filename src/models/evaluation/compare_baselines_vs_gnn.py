from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from src.core.config import load_config, resolve_project_path
from src.graph.data import (
    add_occupation_labels,
    add_train_val_test_masks,
    load_heterodata,
    load_occupation_nodes,
)
from src.models.baselines.majority_class import majority_class_baseline
from src.models.evaluation.build_model import build_model_and_optimizer
from src.models.evaluation.train_model import train_with_early_stopping


RESULTS_OUTPUT_DIR = Path("data/processed/results/soc_classifier")
RESULTS_OUTPUT_PATH = RESULTS_OUTPUT_DIR / "soc_classifier_multiseed_results.csv"
SPLITS = ["train", "val", "test"]


def result_columns(include_balanced_accuracy: bool) -> list[str]:
    """Return the canonical flattened result column order."""

    columns = ["graph_variant", "model_name", "seed"]

    metrics = ["accuracy", "macro_f1"]
    if include_balanced_accuracy:
        metrics.append("balanced_accuracy")

    for split in SPLITS:
        for metric in metrics:
            columns.append(f"{split}_{metric}")

    columns.extend(["best_epoch", "best_val_accuracy"])
    return columns


def build_result_row(
    graph_variant: str,
    seed: int,
    model_name: str,
    results: dict[str, object],
    include_balanced_accuracy: bool,
) -> dict[str, object]:
    """Flatten one model's evaluation output into a CSV-ready row."""

    row: dict[str, object] = {
        "graph_variant": graph_variant,
        "model_name": model_name,
        "seed": seed,
    }

    metrics = ["accuracy", "macro_f1"]
    if include_balanced_accuracy:
        metrics.append("balanced_accuracy")

    for split in SPLITS:
        for metric in metrics:
            key = f"{split}_{metric}"
            row[key] = results.get(key)

    row["best_epoch"] = results.get("best_epoch")
    row["best_val_accuracy"] = results.get("best_val_accuracy")
    return row


def run_seed(
    seed: int,
    graph_variant: str,
    graph_path: Path,
    featured_nodes_dir: Path,
    model_names: list[str],
    include_balanced_accuracy: bool = False,
) -> list[dict[str, object]]:
    """Run one seed and return one flat row per model."""

    print("\n==============================")
    print(f"Running seed {seed}")
    print("==============================")

    data = load_heterodata(graph_path)
    occupation_nodes = load_occupation_nodes(featured_nodes_dir)

    data, label_to_idx, _ = add_occupation_labels(
        data,
        occupation_nodes,
    )
    data = add_train_val_test_masks(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=seed,
    )

    num_classes = len(label_to_idx)
    seed_rows: list[dict[str, object]] = []

    print("\nMajority-class baseline")
    majority_class_results = majority_class_baseline(
        data,
        include_balanced_accuracy=include_balanced_accuracy,
    )
    seed_rows.append(
        build_result_row(
            graph_variant=graph_variant,
            seed=seed,
            model_name="majority_class",
            results=majority_class_results,
            include_balanced_accuracy=include_balanced_accuracy,
        )
    )

    for model_name in model_names:
        print(f"\nTraining {model_name}...")

        model, optimizer, graph_aware, edge_aware = build_model_and_optimizer(
            model_name=model_name,
            data=data,
            num_classes=num_classes,
        )

        training_output = train_with_early_stopping(
            model=model,
            data=data,
            optimizer=optimizer,
            graph_aware=graph_aware,
            edge_aware=edge_aware,
            include_balanced_accuracy=include_balanced_accuracy,
            num_epochs=2500,
            patience=200,
            print_every=100,
        )

        seed_rows.append(
            build_result_row(
                graph_variant=graph_variant,
                seed=seed,
                model_name=model_name,
                results=training_output["results"],
                include_balanced_accuracy=include_balanced_accuracy,
            )
        )

    return seed_rows


def summarize_rows(rows: list[dict[str, object]]) -> None:
    """Print mean and standard deviation for each tracked metric."""

    if not rows:
        print("\nNo rows to summarize.")
        return

    results_df = pd.DataFrame(rows)
    model_names = sorted(results_df["model_name"].dropna().unique().tolist())

    metrics = ["train_accuracy", "val_accuracy", "test_accuracy"]
    metrics.extend(["train_macro_f1", "val_macro_f1", "test_macro_f1"])
    if "train_balanced_accuracy" in results_df.columns:
        metrics.extend(
            [
                "train_balanced_accuracy",
                "val_balanced_accuracy",
                "test_balanced_accuracy",
            ]
        )

    for model_name in model_names:
        model_rows = results_df[results_df["model_name"] == model_name]
        print(f"\n{model_name}")

        for metric in metrics:
            if metric not in model_rows.columns:
                continue

            values = model_rows[metric].dropna().tolist()
            if not values:
                continue

            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            print(f"{metric}: {metric_mean:.3f} ± {metric_std:.3f}")

        best_epochs = model_rows["best_epoch"].dropna().tolist()
        if best_epochs:
            print(f"best_epoch_mean: {mean(best_epochs):.1f}")


def save_results_csv(
    rows: list[dict[str, object]],
    output_path: Path,
    include_balanced_accuracy: bool,
) -> None:
    """Write flattened multi-seed rows to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = result_columns(include_balanced_accuracy=include_balanced_accuracy)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def main() -> None:
    """Run the comparison across graph variants and random seeds."""

    config = load_config()
    processed_graphs_dir = resolve_project_path(
        config["paths"]["processed_graphs_dir"]
    )
    featured_nodes_dir = resolve_project_path(
        config["paths"]["featured_nodes_dir"]
    )

    graph_variants = [
        "dense",
        "core_broad",
        "core_strict",
    ]
    seeds = [0, 1, 2, 3, 4]
    model_names = [
        "mlp",
        "hetero_sage",
        "hetero_transformer",
    ]
    include_balanced_accuracy = True

    all_rows: list[dict[str, object]] = []

    for graph_variant in graph_variants:
        print("\n" + "=" * 40)
        print(f"Running graph variant: {graph_variant}")
        print("=" * 40)

        graph_path = processed_graphs_dir / f"{graph_variant}_heterodata.pt"
        variant_rows: list[dict[str, object]] = []

        for seed in seeds:
            seed_rows = run_seed(
                seed=seed,
                graph_variant=graph_variant,
                graph_path=graph_path,
                featured_nodes_dir=featured_nodes_dir,
                model_names=model_names,
                include_balanced_accuracy=include_balanced_accuracy,
            )
            variant_rows.extend(seed_rows)
            all_rows.extend(seed_rows)

        print("\n" + "=" * 40)
        print(f"Summary for graph variant: {graph_variant}")
        print("=" * 40)
        summarize_rows(variant_rows)

    results_output_path = resolve_project_path(RESULTS_OUTPUT_PATH)
    save_results_csv(
        rows=all_rows,
        output_path=results_output_path,
        include_balanced_accuracy=include_balanced_accuracy,
    )

    print("\n" + "=" * 40)
    print("All graph variant summaries")
    print("=" * 40)
    summarize_rows(all_rows)
    print(f"\nSaved multi-seed results to {results_output_path}")


if __name__ == "__main__":
    main()
