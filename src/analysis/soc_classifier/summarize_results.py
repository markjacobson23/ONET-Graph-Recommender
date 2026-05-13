from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import resolve_project_path


RESULTS_DIR = Path("data/processed/results/soc_classifier")
MULTISEED_RESULTS_FILENAME = "soc_classifier_multiseed_results.csv"
SUMMARY_FILENAME = "soc_classifier_multiseed_summary.csv"
BASELINE_IMPROVEMENTS_FILENAME = "soc_classifier_baseline_improvements.csv"
SEED_WINS_FILENAME = "soc_classifier_seed_wins.csv"

SUMMARY_METRICS = [
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_macro_f1",
    "val_macro_f1",
    "test_macro_f1",
]
BASELINE_MODEL_NAMES = ["mlp", "majority_class"]
SUMMARY_COLUMNS = [
    "graph_variant",
    "model_name",
    "train_accuracy_mean",
    "train_accuracy_std",
    "val_accuracy_mean",
    "val_accuracy_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "train_macro_f1_mean",
    "train_macro_f1_std",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "best_epoch_mean",
]
BASELINE_IMPROVEMENT_COLUMNS = [
    "graph_variant",
    "best_model_name",
    "baseline_model_name",
    "best_test_accuracy_mean",
    "baseline_test_accuracy_mean",
    "test_accuracy_improvement_pp",
    "best_test_macro_f1_mean",
    "baseline_test_macro_f1_mean",
    "test_macro_f1_improvement",
]
SEED_WIN_COLUMNS = ["graph_variant", "model_name", "num_seed_wins"]


def _mean_or_zero(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.mean()) if not values.empty else 0.0


def _std_or_zero(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def load_multiseed_results(results_path: Path) -> pd.DataFrame:
    """Load the flattened per-seed results table."""

    return pd.read_csv(results_path)


def summarize_multiseed_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows into per-variant, per-model summary rows."""

    grouped = results_df.groupby(["graph_variant", "model_name"], sort=True)

    summary_rows: list[dict[str, object]] = []
    for (graph_variant, model_name), group in grouped:
        row: dict[str, object] = {
            "graph_variant": graph_variant,
            "model_name": model_name,
        }

        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = _mean_or_zero(group[metric])
            row[f"{metric}_std"] = _std_or_zero(group[metric])

        row["best_epoch_mean"] = _mean_or_zero(group["best_epoch"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["graph_variant", "model_name"],
            kind="stable",
        ).reset_index(drop=True)
        summary_df = summary_df[SUMMARY_COLUMNS]
    return summary_df


def _pick_best_model(summary_df: pd.DataFrame) -> pd.Series:
    """Pick the best model for a graph variant using the requested tie-break."""

    ordered = summary_df.sort_values(
        by=["test_accuracy_mean", "test_macro_f1_mean", "model_name"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_baseline_improvements(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute best-model comparisons against MLP and majority-class baselines."""

    rows: list[dict[str, object]] = []
    for graph_variant, variant_summary in summary_df.groupby(
        "graph_variant",
        sort=True,
    ):
        best_model = _pick_best_model(variant_summary)
        best_model_name = str(best_model["model_name"])

        for baseline_name in BASELINE_MODEL_NAMES:
            baseline_rows = variant_summary[
                variant_summary["model_name"] == baseline_name
            ]
            if baseline_rows.empty:
                continue

            baseline = baseline_rows.iloc[0]
            rows.append(
                {
                    "graph_variant": graph_variant,
                    "best_model_name": best_model_name,
                    "baseline_model_name": baseline_name,
                    "best_test_accuracy_mean": best_model["test_accuracy_mean"],
                    "baseline_test_accuracy_mean": baseline["test_accuracy_mean"],
                    "test_accuracy_improvement_pp": (
                        best_model["test_accuracy_mean"]
                        - baseline["test_accuracy_mean"]
                    )
                    * 100.0,
                    "best_test_macro_f1_mean": best_model["test_macro_f1_mean"],
                    "baseline_test_macro_f1_mean": baseline["test_macro_f1_mean"],
                    "test_macro_f1_improvement": (
                        best_model["test_macro_f1_mean"]
                        - baseline["test_macro_f1_mean"]
                    ),
                }
            )

    improvements_df = pd.DataFrame(rows)
    if not improvements_df.empty:
        improvements_df = improvements_df.sort_values(
            ["graph_variant", "baseline_model_name"],
            kind="stable",
        ).reset_index(drop=True)
        improvements_df = improvements_df[BASELINE_IMPROVEMENT_COLUMNS]
    return improvements_df


def _pick_seed_winner(seed_rows: pd.DataFrame) -> pd.Series:
    """Pick the winner for one seed using accuracy, F1, and model name tie-breaks."""

    ordered = seed_rows.sort_values(
        by=["test_accuracy", "test_macro_f1", "model_name"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ordered.iloc[0]


def compute_seed_wins(results_df: pd.DataFrame) -> pd.DataFrame:
    """Count how many seed-level wins each model gets per graph variant."""

    winner_rows: list[dict[str, object]] = []

    for graph_variant, variant_rows in results_df.groupby("graph_variant", sort=True):
        wins_by_model: dict[str, int] = {
            model_name: 0
            for model_name in sorted(variant_rows["model_name"].dropna().unique())
        }

        for seed, seed_rows in variant_rows.groupby("seed", sort=True):
            winner = _pick_seed_winner(seed_rows)
            model_name = str(winner["model_name"])
            wins_by_model[model_name] = wins_by_model.get(model_name, 0) + 1

        for model_name, num_seed_wins in sorted(wins_by_model.items()):
            winner_rows.append(
                {
                    "graph_variant": graph_variant,
                    "model_name": model_name,
                    "num_seed_wins": num_seed_wins,
                }
            )

    seed_wins_df = pd.DataFrame(winner_rows)
    if not seed_wins_df.empty:
        seed_wins_df = seed_wins_df.sort_values(
            ["graph_variant", "num_seed_wins", "model_name"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        seed_wins_df = seed_wins_df[SEED_WIN_COLUMNS]
    return seed_wins_df


def write_analysis_outputs(
    results_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate and save all summary CSVs from the per-seed results table."""

    output_dir = resolve_project_path(output_dir or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = load_multiseed_results(results_path)
    summary_df = summarize_multiseed_results(results_df)
    baseline_improvements_df = build_baseline_improvements(summary_df)
    seed_wins_df = compute_seed_wins(results_df)

    summary_path = output_dir / SUMMARY_FILENAME
    baseline_improvements_path = output_dir / BASELINE_IMPROVEMENTS_FILENAME
    seed_wins_path = output_dir / SEED_WINS_FILENAME

    summary_df.to_csv(summary_path, index=False)
    baseline_improvements_df.to_csv(baseline_improvements_path, index=False)
    seed_wins_df.to_csv(seed_wins_path, index=False)

    return {
        "summary": summary_path,
        "baseline_improvements": baseline_improvements_path,
        "seed_wins": seed_wins_path,
    }


def main() -> None:
    """Generate SOC classifier summary CSVs from the saved per-seed results."""

    results_path = resolve_project_path(RESULTS_DIR / MULTISEED_RESULTS_FILENAME)
    output_paths = write_analysis_outputs(results_path)
    print(f"Wrote summary CSVs to {output_paths['summary'].parent}")


if __name__ == "__main__":
    main()
