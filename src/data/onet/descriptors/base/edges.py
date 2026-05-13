from __future__ import annotations

import pandas as pd


def build_occupation_descriptor_edges(
    occupation_descriptor_edge_rows: pd.DataFrame,
    occupation_nodes: pd.DataFrame,
    descriptor_nodes: pd.DataFrame,
    descriptor_config: dict,
) -> pd.DataFrame:
    """Merge raw edge rows with node indices and sort the result."""

    id_col = descriptor_config["id_col"]
    idx_col = descriptor_config["idx_col"]

    # Join occupation ids first so every edge points at the shared occupation index.
    occupation_descriptor_edges = occupation_descriptor_edge_rows.merge(
        occupation_nodes[["onetsoc_code", "occupation_idx"]],
        on="onetsoc_code",
        how="left",
    )

    # Then attach the descriptor node index.
    occupation_descriptor_edges = occupation_descriptor_edges.merge(
        descriptor_nodes[[id_col, idx_col]],
        left_on="descriptor_id",
        right_on=id_col,
        how="left",
    )

    occupation_descriptor_edges = occupation_descriptor_edges[
        ["occupation_idx", idx_col, "importance", "level"]
    ]

    return (
        occupation_descriptor_edges
        .sort_values(by=["occupation_idx", idx_col])
        .reset_index(drop=True)
    )
