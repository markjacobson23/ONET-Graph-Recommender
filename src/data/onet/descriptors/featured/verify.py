import pandas as pd


def verify_featured_nodes(base_nodes: pd.DataFrame, featured_nodes: pd.DataFrame, node_schema: dict):

    """Verify that the featured nodes table has a valid number of rows,
      no missing values, and proper feature columns."""

    idx_col = node_schema["idx_col"]
    metadata_cols = node_schema["metadata_cols"]

    assert len(featured_nodes) == len(base_nodes), (
        "Node row count changed after adding features"
    )

    assert featured_nodes[idx_col].notna().all(), (
        f"Missing {idx_col} values in node table"
    )

    assert set(featured_nodes[idx_col]) == set(base_nodes[idx_col]), (
        f"{idx_col} values changed after adding features"
    )

    feature_cols = [
        col for col in featured_nodes.columns
        if col not in metadata_cols
    ]

    assert len(feature_cols) > 0, (
        "No feature columns found"
    )

    assert featured_nodes[feature_cols].notna().all().all(), (
        "Missing numeric feature values in node table"
    )

    assert all(
        featured_nodes[col].dtype.kind in "iuf"
        for col in feature_cols
    ), "Some feature columns are not numeric"
