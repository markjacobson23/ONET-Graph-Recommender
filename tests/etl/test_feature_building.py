from __future__ import annotations

import math

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.onet.descriptors.featured.features import (
    attach_features_to_nodes,
    build_descriptor_features,
    build_occupation_descriptor_features,
    fill_missing_feature_values,
)


def test_build_occupation_descriptor_features_exact_values(
    tiny_occupation_skill_edges: pd.DataFrame,
    skill_config: dict,
) -> None:
    result = build_occupation_descriptor_features(
        tiny_occupation_skill_edges,
        skill_config,
    ).sort_values("occupation_idx").reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "occupation_idx": [0, 1],
            "avg_skill_importance": [3.0, 5.0],
            "avg_skill_level": [4.0, 6.0],
            "max_skill_importance": [4.0, 5.0],
            "max_skill_level": [5.0, 6.0],
            "min_skill_importance": [2.0, 5.0],
            "min_skill_level": [3.0, 6.0],
            "std_skill_importance": [math.sqrt(2.0), math.nan],
            "std_skill_level": [math.sqrt(2.0), math.nan],
            "num_high_importance_skills": [1, 1],
            "num_high_level_skills": [1, 1],
            "num_core_skills": [1, 1],
        }
    )

    assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_descriptor_features_groups_by_descriptor_index(
    tiny_occupation_skill_edges: pd.DataFrame,
    skill_config: dict,
) -> None:
    result = build_descriptor_features(
        tiny_occupation_skill_edges,
        skill_config,
    ).sort_values("skill_idx").reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "skill_idx": [0, 1],
            "avg_importance": [4.5, 2.0],
            "avg_level": [5.5, 3.0],
            "max_importance": [5.0, 2.0],
            "max_level": [6.0, 3.0],
            "min_importance": [4.0, 2.0],
            "min_level": [5.0, 3.0],
            "std_importance": [math.sqrt(0.5), math.nan],
            "std_level": [math.sqrt(0.5), math.nan],
            "num_high_importance_occupations": [2.0, math.nan],
            "num_high_level_occupations": [2.0, math.nan],
            "num_core_occupations": [2.0, math.nan],
        }
    )

    assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_attach_features_to_nodes_preserves_row_count(
    tiny_occupation_nodes: pd.DataFrame,
) -> None:
    features = pd.DataFrame(
        {
            "occupation_idx": [0, 1],
            "avg_skill_importance": [4.0, 5.0],
        }
    )

    result = attach_features_to_nodes(
        tiny_occupation_nodes,
        features,
        on_col="occupation_idx",
    )

    assert len(result) == len(tiny_occupation_nodes)


def test_fill_missing_feature_values_only_fills_feature_columns(
    tiny_node_table_with_missing_values: pd.DataFrame,
) -> None:
    node_schema = {
        "metadata_cols": [
            "occupation_idx",
            "onetsoc_code",
            "occupation_title",
        ]
    }

    result = fill_missing_feature_values(
        tiny_node_table_with_missing_values,
        node_schema["metadata_cols"],
    )

    assert result["occupation_idx"].tolist() == [0, 1]
    assert result["onetsoc_code"].tolist() == ["15-2051.00", "17-2141.02"]
    assert result["occupation_title"].tolist() == ["Data Scientist", "Engineer"]
    assert result["avg_skill_importance"].tolist() == [4.5, 0.0]
    assert result["avg_skill_level"].tolist() == [5.0, 0.0]
