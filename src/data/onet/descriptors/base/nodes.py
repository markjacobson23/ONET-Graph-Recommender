from __future__ import annotations

import pandas as pd


def build_occupation_nodes(occupation_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the occupation node table with stable integer indices."""

    occupation_nodes = (
        occupation_rows[["onetsoc_code", "occupation_title"]]
        .drop_duplicates()
        .sort_values("onetsoc_code")
        .reset_index(drop=True)
    )

    # Occupation indices are shared across every descriptor edge table.
    occupation_nodes["occupation_idx"] = occupation_nodes.index

    return occupation_nodes[["occupation_idx", "onetsoc_code", "occupation_title"]]


def build_descriptor_nodes(
    descriptor_rows: pd.DataFrame,
    descriptor_config: dict,
) -> pd.DataFrame:
    """Build a descriptor node table using the config-defined column names."""

    descriptor_nodes = (
        descriptor_rows[["descriptor_id", "descriptor_name"]]
        .drop_duplicates()
        .sort_values("descriptor_id")
        .reset_index(drop=True)
    )

    # Descriptor indices are local to their node type.
    descriptor_nodes["descriptor_idx"] = descriptor_nodes.index

    descriptor_nodes = descriptor_nodes[
        ["descriptor_idx", "descriptor_id", "descriptor_name"]
    ]

    return descriptor_nodes.rename(
        columns={
            "descriptor_idx": descriptor_config["idx_col"],
            "descriptor_id": descriptor_config["id_col"],
            "descriptor_name": descriptor_config["name_col"],
        }
    )
