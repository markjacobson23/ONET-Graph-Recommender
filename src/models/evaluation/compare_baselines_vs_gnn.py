from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

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


def run_seed(
    seed: int,
    graph_path: Path,
    featured_nodes_dir: Path,
    model_names: list[str],
) -> dict[str, dict[str, object]]:
    """Run one seed for the baseline and GNN comparison."""

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
    seed_results: dict[str, dict] = {}

    print("\nMajority-class baseline")
    seed_results["majority_class"] = majority_class_baseline(data)

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
            num_epochs=2500,
            patience=200,
            print_every=100,
        )

        seed_results[model_name] = training_output["results"]

    return seed_results


def summarize_seed_results(
    all_results: dict[int, dict[str, dict[str, object]]],
) -> None:
    """Print mean and standard deviation for each tracked metric."""

    model_names = sorted(
        {
            model_name
            for seed_results in all_results.values()
            for model_name in seed_results.keys()
        }
    )

    metrics = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
    ]

    print("\n\n==============================")
    print("Multi-seed summary")
    print("==============================")

    for model_name in model_names:
        print(f"\n{model_name}")

        for metric in metrics:
            values = [
                seed_results[model_name][metric]
                for seed_results in all_results.values()
            ]

            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            print(f"{metric}: {metric_mean:.3f} ± {metric_std:.3f}")

        best_epochs = [
            seed_results[model_name].get("best_epoch")
            for seed_results in all_results.values()
            if "best_epoch" in seed_results[model_name]
        ]

        if best_epochs:
            print(f"best_epoch_mean: {mean(best_epochs):.1f}")


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

    all_variant_results = {}

    for graph_variant in graph_variants:
        print("\n" + "=" * 40)
        print(f"Running graph variant: {graph_variant}")
        print("=" * 40)

        graph_path = processed_graphs_dir / f"{graph_variant}_heterodata.pt"
        all_seed_results = {}

        for seed in seeds:
            all_seed_results[seed] = run_seed(
                seed=seed,
                graph_path=graph_path,
                featured_nodes_dir=featured_nodes_dir,
                model_names=model_names,
            )

        all_variant_results[graph_variant] = all_seed_results

        print("\n" + "=" * 40)
        print(f"Summary for graph variant: {graph_variant}")
        print("=" * 40)
        summarize_seed_results(all_seed_results)

    print("\n" + "=" * 40)
    print("All graph variant summaries")
    print("=" * 40)

    for graph_variant, variant_results in all_variant_results.items():
        print("\n" + "-" * 40)
        print(f"{graph_variant}")
        print("-" * 40)
        summarize_seed_results(variant_results)


if __name__ == "__main__":
    main()
