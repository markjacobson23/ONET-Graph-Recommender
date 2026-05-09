import pandas as pd
from pathlib import Path

def build_occupation_descriptor_edges(
    occupation_descriptor_edge_rows,
    occupation_nodes,
    descriptor_nodes,
    descriptor_config,
):
    id_col = descriptor_config["id_col"]
    idx_col = descriptor_config["idx_col"]

    occupation_descriptor_edges = occupation_descriptor_edge_rows.merge(
        occupation_nodes[["onetsoc_code", "occupation_idx"]],
        on="onetsoc_code",
        how="left"
    )

    occupation_descriptor_edges = occupation_descriptor_edges.merge(
        descriptor_nodes[[id_col, idx_col]],
        left_on="descriptor_id",
        right_on=id_col,
        how="left"
    )

    occupation_descriptor_edges = occupation_descriptor_edges[
        ["occupation_idx", idx_col, "importance", "level"]
    ]

    occupation_descriptor_edges = (
        occupation_descriptor_edges
        .sort_values(by=["occupation_idx", idx_col])
        .reset_index(drop=True)
    )

    return occupation_descriptor_edges
