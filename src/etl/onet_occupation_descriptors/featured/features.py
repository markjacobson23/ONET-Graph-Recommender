import pandas as pd


def build_occupation_descriptor_features(edges: pd.DataFrame, descriptor_config: dict) -> pd.DataFrame:

    """build basic aggregated features for the occupation nodes relative to the descriptor type.
       features:
        - avg_descriptor_importance,
        - avg_descriptor_level,
        - max_descriptor_importance,
        - max_descriptor_level,
        - min_descriptor_importance,
        - min_descriptor_level,
        - std_descriptor_importance,
        - std_descriptor_level,
        - num_high_importance_descriptors, <- this is the number of unique descriptors with importance >= 4.0
        - num_high_level_descriptors, <- this is the number of unique descriptors with level >= 5.0
        - num_core_descriptors, <- this is the number of unique descriptors with importance >= 4.0 AND level >= 5.0
          """

    # get the idx column name, feature_prefix, and count relative name suffix from the descriptor config
    idx_col = descriptor_config["idx_col"]
    feature_count_name = descriptor_config["feature_count_name"]
    feature_prefix = descriptor_config["feature_prefix"]

    # build basic aggregated features for occupation-descriptor edges table
    agg_map = {
        f"avg_{feature_prefix}_importance": ("importance", "mean"),
        f"avg_{feature_prefix}_level": ("level", "mean"),
        f"max_{feature_prefix}_importance": ("importance", "max"),
        f"max_{feature_prefix}_level": ("level", "max"),
        f"min_{feature_prefix}_importance": ("importance", "min"),
        f"min_{feature_prefix}_level": ("level", "min"),
        f"std_{feature_prefix}_importance": ("importance", "std"),
        f"std_{feature_prefix}_level": ("level", "std"),
    }
    basic_occupation_descriptor_features = (
        edges
        .groupby("occupation_idx")
        .agg(**agg_map)
        .reset_index()
    )

    # number of unique descriptors with importance >= 4.0
    high_importance_col = f"num_high_importance_{feature_count_name}"
    high_importance_counts = (
        edges[edges["importance"] >= 4.0]
        .groupby("occupation_idx")
        .agg(**{high_importance_col: (idx_col, "nunique")})
        .reset_index()
    )

    # number of unique descriptors with level >= 5.0
    high_level_col = f"num_high_level_{feature_count_name}"
    high_level_counts = (
        edges[edges["level"] >= 5.0]
        .groupby("occupation_idx")
        .agg(**{high_level_col: (idx_col, "nunique")})
        .reset_index()
    )

    # number of unique descriptors with importance >= 4.0 AND level >= 5.0
    core_col = f"num_core_{feature_count_name}"
    core_counts = (
        edges[
            (edges["importance"] >= 4.0)
            & (edges["level"] >= 5.0)
            ]
        .groupby("occupation_idx")
        .agg(**{core_col: (idx_col, "nunique")})
        .reset_index()
    )

    # merge all the features together
    all_occupation_descriptor_features = (
        basic_occupation_descriptor_features
        .merge(high_importance_counts, on="occupation_idx", how="left")
        .merge(high_level_counts, on="occupation_idx", how="left")
        .merge(core_counts, on="occupation_idx", how="left")
    )

    return all_occupation_descriptor_features


def build_descriptor_features(edges: pd.DataFrame, descriptor_config: dict) -> pd.DataFrame:

    """build basic aggregated features for the descriptor nodes relative to the occupation.
       features:
        - avg_importance,
        - avg_level,
        - max_importance,
        - max_level,
        - min_importance,
        - min_level,
        - std_importance,
        - std_level,
        - num_high_importance_occupations, <- this is the number of unique occupations with this descriptor importance >= 4.0
        - num_high_level_occupations, <- this is the number of unique occupations with this descriptor level >= 5.0
        - num_core_occupations, <- this is the number of unique occupations with descriptor importance >= 4.0 AND descriptor level >= 5.0
          """

    # get the idx column name from the descriptor config
    idx_col = descriptor_config["idx_col"]

    # build basic aggregated features for the descriptor nodes table
    basic_descriptor_features = (
        edges
        .groupby(idx_col)
        .agg(
            avg_importance=("importance", "mean"),
            avg_level=("level", "mean"),
            max_importance=("importance", "max"),
            max_level=("level", "max"),
            min_importance=("importance", "min"),
            min_level=("level", "min"),
            std_importance=("importance", "std"),
            std_level=("level", "std"),
            )
        .reset_index()
    )

    # number of unique occupations with this descriptor importance >= 4.0
    high_importance_occupation_counts = (
        edges[edges["importance"] >= 4.0]
        .groupby(idx_col)
        .agg(num_high_importance_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    # number of unique occupations with this descriptor level >= 5.0
    high_level_occupation_counts = (
        edges[edges["level"] >= 5.0]
        .groupby(idx_col)
        .agg(num_high_level_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    # number of unique occupations with this descriptor importance >= 4.0 AND this descriptor level >= 5.0
    core_occupation_counts = (
        edges[
            (edges["importance"] >= 4.0)
            & (edges["level"] >= 5.0)
            ]
        .groupby(idx_col)
        .agg(num_core_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    # merge all the features together
    all_descriptor_features = (
        basic_descriptor_features
        .merge(high_importance_occupation_counts, on=idx_col, how="left")
        .merge(high_level_occupation_counts, on=idx_col, how="left")
        .merge(core_occupation_counts, on=idx_col, how="left")
    )

    return all_descriptor_features

def attach_features_to_nodes(nodes: pd.DataFrame, features: pd.DataFrame, on_col: str) -> pd.DataFrame:
    """attach the features to the node table"""
    featured_nodes = pd.merge(
        nodes,
        features,
        on=on_col,
        how="left"
    )
    return featured_nodes

def fill_missing_feature_values(node_table: pd.DataFrame, metadata_cols: list[str]) -> pd.DataFrame:
    """fill missing feature values with 0"""

    # make a copy of the node table to avoid modifying the original
    node_table = node_table.copy()

    # get the feature columns
    feature_cols = [
        col for col in node_table.columns
        if col not in metadata_cols
    ]

    # fill missing values with 0 (this is fine since torch casts to float)
    node_table[feature_cols] = node_table[feature_cols].fillna(0)

    return node_table
