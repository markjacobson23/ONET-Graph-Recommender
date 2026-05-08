import pandas as pd
from pathlib import Path
from src.utils.config import load_config, resolve_project_path

def load_base_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupation_nodes = pd.read_csv(input_dir / "occupation_nodes.csv")
    skill_nodes = pd.read_csv(input_dir / "skill_nodes.csv")
    occupation_skill_edges = pd.read_csv(input_dir / "occupation_skill_edges.csv")
    return occupation_nodes, skill_nodes, occupation_skill_edges

def build_occupation_features(occupation_skill_edges):
    occupation_features = (
        occupation_skill_edges
        .groupby("occupation_idx")
        .agg(
            avg_skill_importance=("importance", "mean"),
            avg_skill_level=("level", "mean"),
            max_skill_importance=("importance", "max"),
            max_skill_level=("level", "max"),
            min_skill_importance=("importance", "min"),
            min_skill_level=("level", "min"),
            std_skill_importance=("importance", "std"),
            std_skill_level=("level", "std"),
            )
        .reset_index()
    )

    high_importance_counts = (
        occupation_skill_edges[occupation_skill_edges["importance"] >= 4.0]
        .groupby("occupation_idx")
        .agg(num_high_importance_skills=("skill_idx", "nunique"))
        .reset_index()
    )

    occupation_features = pd.merge(
        occupation_features,
        high_importance_counts,
        on="occupation_idx",
        how="left"
    )

    high_level_counts = (
        occupation_skill_edges[occupation_skill_edges["level"] >= 5.0]
        .groupby("occupation_idx")
        .agg(num_high_level_skills=("skill_idx", "nunique"))
        .reset_index()
    )

    occupation_features = pd.merge(
        occupation_features,
        high_level_counts,
        on="occupation_idx",
        how="left"
    )

    core_skill_counts = (
        occupation_skill_edges[
            (occupation_skill_edges["importance"] >= 4.0)
            & (occupation_skill_edges["level"] >= 5.0)
            ]
        .groupby("occupation_idx")
        .agg(num_core_skills=("skill_idx", "nunique"))
        .reset_index()
    )

    occupation_features = pd.merge(
        occupation_features,
        core_skill_counts,
        on="occupation_idx",
        how="left"
    )

    return occupation_features

def build_skill_features(occupation_skill_edges):
    skill_features = (
        occupation_skill_edges
        .groupby("skill_idx")
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
        occupation_skill_edges[occupation_skill_edges["importance"] >= 4.0]
        .groupby("skill_idx")
        .agg(num_high_importance_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    skill_features = pd.merge(
        skill_features,
        high_importance_occupation_counts,
        on="skill_idx",
        how="left"
    )

    high_level_occupation_counts = (
        occupation_skill_edges[occupation_skill_edges["level"] >= 5.0]
        .groupby("skill_idx")
        .agg(num_high_level_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    skill_features = pd.merge(
        skill_features,
        high_level_occupation_counts,
        on="skill_idx",
        how="left"
    )

    core_occupation_counts = (
        occupation_skill_edges[
            (occupation_skill_edges["importance"] >= 4.0)
            & (occupation_skill_edges["level"] >= 5.0)
            ]
        .groupby("skill_idx")
        .agg(num_core_occupations=("occupation_idx", "nunique"))
        .reset_index()
    )

    skill_features = pd.merge(
        skill_features,
        core_occupation_counts,
        on="skill_idx",
        how="left"
    )

    return skill_features

def attach_occupation_features(occupation_features, occupation_nodes):
    featured_occupation_nodes = pd.merge(
        occupation_nodes,
        occupation_features,
        on="occupation_idx",
        how="left"
    )
    return featured_occupation_nodes

def attach_skill_features(skill_features, skill_nodes):
    featured_skill_nodes = pd.merge(
        skill_nodes,
        skill_features,
        on="skill_idx",
        how="left"
    )
    return featured_skill_nodes

def fill_missing_feature_values(featured_occupation_nodes, featured_skill_nodes):

    occ_float_cols = featured_occupation_nodes.select_dtypes(include=['float']).columns
    occ_int_cols = featured_occupation_nodes.select_dtypes(include=['int']).columns
    skill_float_cols = featured_skill_nodes.select_dtypes(include=['float']).columns
    skill_int_cols = featured_skill_nodes.select_dtypes(include=['int']).columns

    featured_occupation_nodes[occ_float_cols] = featured_occupation_nodes[occ_float_cols].fillna(0.0)
    featured_occupation_nodes[occ_int_cols] = featured_occupation_nodes[occ_int_cols].fillna(0)

    featured_skill_nodes[skill_float_cols] = featured_skill_nodes[skill_float_cols].fillna(0.0)
    featured_skill_nodes[skill_int_cols] = featured_skill_nodes[skill_int_cols].fillna(0)

    return featured_occupation_nodes, featured_skill_nodes

def verify_feature_tables(
    base_occupation_nodes,
    base_skill_nodes,
    featured_occupation_nodes,
    featured_skill_nodes,
    occupation_skill_edges,
):
    # check that the row counts of the tables are the same after adding features
    assert len(featured_occupation_nodes) == len(base_occupation_nodes), (
        "Occupation node row count changed after adding features"
    )
    assert len(featured_skill_nodes) == len(base_skill_nodes), (
        "Skill node row count changed after adding features"
    )

    # check that all occupation_idx and skill_idx values are present in the tables
    assert featured_occupation_nodes["occupation_idx"].notna().all(), (
        "Missing occupation_idx values in occupation node table"
    )
    assert featured_skill_nodes["skill_idx"].notna().all(), (
        "Missing skill_idx values in skill node table"
    )

    # check that all occupation_idx and skill_idx values are present in the edge table
    assert occupation_skill_edges["occupation_idx"].notna().all(), (
        "Missing occupation_idx values in edge table"
    )
    assert occupation_skill_edges["skill_idx"].notna().all(), (
        "Missing skill_idx values in edge table"
    )

    # check that all importance and level values are present in the edge table
    assert occupation_skill_edges["importance"].notna().all(), (
        "Missing importance values in edge table"
    )
    assert occupation_skill_edges["level"].notna().all(), (
        "Missing level values in edge table"
    )

    # check that the importance and level values are within the expected range
    assert occupation_skill_edges["importance"].between(0, 5).all(), (
        "Importance values outside expected range [0, 5]"
    )
    assert occupation_skill_edges["level"].between(0, 7).all(), (
        "Level values outside expected range [0, 7]"
    )

    occupation_metadata_cols = {
        "occupation_idx",
        "onetsoc_code",
        "occupation_title",
    }

    skill_metadata_cols = {
        "skill_idx",
        "skill_id",
        "skill_name",
    }

    occupation_feature_cols = [
        col for col in featured_occupation_nodes.columns
        if col not in occupation_metadata_cols
    ]

    skill_feature_cols = [
        col for col in featured_skill_nodes.columns
        if col not in skill_metadata_cols
    ]

    # check that the feature columns are present in the tables
    assert len(occupation_feature_cols) > 0, (
        "No occupation feature columns found"
    )
    assert len(skill_feature_cols) > 0, (
        "No skill feature columns found"
    )

    # check that the feature columns have values
    assert featured_occupation_nodes[occupation_feature_cols].notna().all().all(), (
        "Missing numeric feature values in occupation node table"
    )
    assert featured_skill_nodes[skill_feature_cols].notna().all().all(), (
        "Missing numeric feature values in skill node table"
    )

    # check that the feature columns have the correct data types
    assert all(
        featured_occupation_nodes[col].dtype.kind in "iuf"
        for col in occupation_feature_cols
    ), "Some occupation feature columns are not numeric"
    assert all(
        featured_skill_nodes[col].dtype.kind in "iuf"
        for col in skill_feature_cols
    ), "Some skill feature columns are not numeric"

def save_featured_tables(
        featured_occupation_nodes,
        featured_skill_nodes,
        occupation_skill_edges,
        output_dir
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    featured_occupation_nodes.to_csv(output_dir / "featured_occupation_nodes.csv", index=False)
    featured_skill_nodes.to_csv(output_dir / "featured_skill_nodes.csv", index=False)
    occupation_skill_edges.to_csv(output_dir / "occupation_skill_edges.csv", index=False)

def main():

    # load config dict
    config = load_config()

    # resolve paths
    input_dir = resolve_project_path(config["paths"]["base_tables_dir"])
    output_dir = resolve_project_path(config["paths"]["featured_tables_dir"])

    # load base tables
    occupation_nodes, skill_nodes, occupation_skill_edges = load_base_tables(input_dir)

    # build occupation features
    occupation_features = build_occupation_features(occupation_skill_edges)

    # build skill features
    skill_features = build_skill_features(occupation_skill_edges)

    # attach occupation features to occupation nodes
    featured_occupation_nodes = attach_occupation_features(occupation_features, occupation_nodes)

    # attach skill features to skill nodes
    featured_skill_nodes = attach_skill_features(skill_features, skill_nodes)

    # fill missing feature values with 0
    featured_occupation_nodes, featured_skill_nodes = (
        fill_missing_feature_values(
            featured_occupation_nodes,
            featured_skill_nodes
        )
    )

    # verify that the tables are as expected
    verify_feature_tables(
        occupation_nodes,
        skill_nodes,
        featured_occupation_nodes,
        featured_skill_nodes,
        occupation_skill_edges
    )

    # save the tables
    save_featured_tables(
        featured_occupation_nodes,
        featured_skill_nodes,
        occupation_skill_edges,
        output_dir)

    print("Featured tables saved successfully.")

if __name__ == "__main__":
    main()