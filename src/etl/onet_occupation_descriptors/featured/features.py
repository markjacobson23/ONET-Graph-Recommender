import pandas as pd


def build_occupation_descriptor_features(edges, descriptor_config):

    idx_col = descriptor_config["idx_col"]
    feature_count_name = descriptor_config["feature_count_name"]
    feature_prefix = descriptor_config["feature_prefix"]

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

    high_importance_col = f"num_high_importance_{feature_count_name}"
    high_importance_counts = (
        edges[edges["importance"] >= 4.0]
        .groupby("occupation_idx")
        .agg(**{high_importance_col: (idx_col, "nunique")})
        .reset_index()
    )

    high_level_col = f"num_high_level_{feature_count_name}"
    high_level_counts = (
        edges[edges["level"] >= 5.0]
        .groupby("occupation_idx")
        .agg(**{high_level_col: (idx_col, "nunique")})
        .reset_index()
    )
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

    all_occupation_descriptor_features = (
        basic_occupation_descriptor_features
        .merge(high_importance_counts, on="occupation_idx", how="left")
        .merge(high_level_counts, on="occupation_idx", how="left")
        .merge(core_counts, on="occupation_idx", how="left")
    )

    return all_occupation_descriptor_features


def build_descriptor_features(edges, descriptor_config):

    idx_col = descriptor_config["idx_col"]

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

    high_importance_occupation_counts = (
        edges[edges["importance"] >= 4.0]
        .groupby(idx_col)
        .agg(num_high_importance_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    high_level_occupation_counts = (
        edges[edges["level"] >= 5.0]
        .groupby(idx_col)
        .agg(num_high_level_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    core_occupation_counts = (
        edges[
            (edges["importance"] >= 4.0)
            & (edges["level"] >= 5.0)
            ]
        .groupby(idx_col)
        .agg(num_core_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    all_descriptor_features = (
        basic_descriptor_features
        .merge(high_importance_occupation_counts, on=idx_col, how="left")
        .merge(high_level_occupation_counts, on=idx_col, how="left")
        .merge(core_occupation_counts, on=idx_col, how="left")
    )

    return all_descriptor_features

def attach_features_to_nodes(nodes, features, on_col):
    featured_nodes = pd.merge(
        nodes,
        features,
        on=on_col,
        how="left"
    )
    return featured_nodes

def fill_missing_feature_values(node_table, metadata_cols):

    node_table = node_table.copy()

    feature_cols = [
        col for col in node_table.columns
        if col not in metadata_cols
    ]

    float_cols = node_table[feature_cols].select_dtypes(include=["float"]).columns
    int_cols = node_table[feature_cols].select_dtypes(include=["int"]).columns

    node_table[feature_cols] = node_table[feature_cols].fillna(0)

    return node_table
