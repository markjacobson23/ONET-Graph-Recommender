from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.soc_classifier import summarize_results as summarize


def _make_row(
    graph_variant: str,
    model_name: str,
    seed: int,
    train_accuracy: float,
    val_accuracy: float,
    test_accuracy: float,
    train_macro_f1: float,
    val_macro_f1: float,
    test_macro_f1: float,
    best_epoch,
) -> dict[str, object]:
    return {
        "graph_variant": graph_variant,
        "model_name": model_name,
        "seed": seed,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
        "best_epoch": best_epoch,
        "best_val_accuracy": val_accuracy,
    }


def _build_results_df() -> pd.DataFrame:
    rows = [
        _make_row("dense", "majority_class", 0, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, None),
        _make_row("dense", "mlp", 0, 0.40, 0.45, 0.50, 0.60, 0.62, 0.60, 10),
        _make_row("dense", "hetero_sage", 0, 0.42, 0.46, 0.55, 0.61, 0.63, 0.62, 11),
        _make_row(
            "dense",
            "hetero_transformer",
            0,
            0.44,
            0.47,
            0.60,
            0.63,
            0.64,
            0.63,
            12,
        ),
        _make_row("dense", "majority_class", 1, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, None),
        _make_row("dense", "mlp", 1, 0.41, 0.46, 0.55, 0.61, 0.63, 0.61, 14),
        _make_row("dense", "hetero_sage", 1, 0.43, 0.47, 0.50, 0.62, 0.65, 0.75, 15),
        _make_row(
            "dense",
            "hetero_transformer",
            1,
            0.45,
            0.48,
            0.54,
            0.64,
            0.66,
            0.64,
            16,
        ),
        _make_row("core_broad", "majority_class", 0, 0.20, 0.20, 0.20, 0.10, 0.10, 0.10, None),
        _make_row("core_broad", "mlp", 0, 0.55, 0.58, 0.60, 0.61, 0.62, 0.61, 9),
        _make_row("core_broad", "hetero_sage", 0, 0.54, 0.57, 0.55, 0.58, 0.59, 0.58, 8),
        _make_row(
            "core_broad",
            "hetero_transformer",
            0,
            0.53,
            0.56,
            0.58,
            0.60,
            0.61,
            0.60,
            7,
        ),
        _make_row("core_broad", "majority_class", 1, 0.25, 0.25, 0.25, 0.10, 0.10, 0.10, None),
        _make_row("core_broad", "mlp", 1, 0.56, 0.59, 0.65, 0.70, 0.71, 0.70, 13),
        _make_row("core_broad", "hetero_sage", 1, 0.55, 0.58, 0.64, 0.69, 0.70, 0.69, 12),
        _make_row(
            "core_broad",
            "hetero_transformer",
            1,
            0.57,
            0.60,
            0.65,
            0.79,
            0.80,
            0.80,
            14,
        ),
    ]
    return pd.DataFrame(rows)


def test_summarize_multiseed_results() -> None:
    results_df = _build_results_df()

    summary_df = summarize.summarize_multiseed_results(results_df)

    assert list(summary_df.columns) == summarize.SUMMARY_COLUMNS
    assert len(summary_df) == 8

    dense_transformer = summary_df[
        (summary_df["graph_variant"] == "dense")
        & (summary_df["model_name"] == "hetero_transformer")
    ].iloc[0]
    assert dense_transformer["test_accuracy_mean"] == pytest.approx(0.57)
    assert dense_transformer["test_accuracy_std"] == pytest.approx(0.04242640687)
    assert dense_transformer["best_epoch_mean"] == pytest.approx(14.0)

    dense_majority = summary_df[
        (summary_df["graph_variant"] == "dense")
        & (summary_df["model_name"] == "majority_class")
    ].iloc[0]
    assert dense_majority["best_epoch_mean"] == pytest.approx(0.0)


def test_baseline_improvements_use_best_model_and_percentage_points() -> None:
    results_df = _build_results_df()
    summary_df = summarize.summarize_multiseed_results(results_df)

    improvements_df = summarize.build_baseline_improvements(summary_df)

    assert list(improvements_df.columns) == summarize.BASELINE_IMPROVEMENT_COLUMNS
    assert len(improvements_df) == 4

    dense_mlp = improvements_df[
        (improvements_df["graph_variant"] == "dense")
        & (improvements_df["baseline_model_name"] == "mlp")
    ].iloc[0]
    assert dense_mlp["best_model_name"] == "hetero_transformer"
    assert dense_mlp["best_test_accuracy_mean"] == pytest.approx(0.57)
    assert dense_mlp["baseline_test_accuracy_mean"] == pytest.approx(0.525)
    assert dense_mlp["test_accuracy_improvement_pp"] == pytest.approx(4.5)
    assert dense_mlp["test_macro_f1_improvement"] == pytest.approx(0.03)

    core_majority = improvements_df[
        (improvements_df["graph_variant"] == "core_broad")
        & (improvements_df["baseline_model_name"] == "majority_class")
    ].iloc[0]
    assert core_majority["best_model_name"] == "mlp"
    assert core_majority["test_accuracy_improvement_pp"] == pytest.approx(40.0)


def test_seed_wins_counts_and_tie_breaks() -> None:
    results_df = _build_results_df()

    seed_wins_df = summarize.compute_seed_wins(results_df)

    assert list(seed_wins_df.columns) == summarize.SEED_WIN_COLUMNS

    dense_wins = seed_wins_df[seed_wins_df["graph_variant"] == "dense"]
    assert dense_wins.set_index("model_name")["num_seed_wins"].to_dict() == {
        "hetero_sage": 0,
        "hetero_transformer": 1,
        "majority_class": 0,
        "mlp": 1,
    }

    core_wins = seed_wins_df[seed_wins_df["graph_variant"] == "core_broad"]
    assert core_wins.set_index("model_name")["num_seed_wins"].to_dict() == {
        "hetero_sage": 0,
        "hetero_transformer": 1,
        "majority_class": 0,
        "mlp": 1,
    }


def test_write_analysis_outputs(tmp_path: Path) -> None:
    results_df = _build_results_df()
    input_path = tmp_path / "results" / "soc_classifier" / "soc_classifier_multiseed_results.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(input_path, index=False)

    output_dir = tmp_path / "results" / "soc_classifier"
    output_paths = summarize.write_analysis_outputs(
        results_path=input_path,
        output_dir=output_dir,
    )

    assert output_paths == {
        "summary": output_dir / summarize.SUMMARY_FILENAME,
        "baseline_improvements": output_dir / summarize.BASELINE_IMPROVEMENTS_FILENAME,
        "seed_wins": output_dir / summarize.SEED_WINS_FILENAME,
    }

    for path in output_paths.values():
        assert path.exists()
