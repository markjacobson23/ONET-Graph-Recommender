import pandas as pd
from pathlib import Path

def build_occupation_nodes(occupation_rows):
    occupation_nodes = (
        occupation_rows[["onetsoc_code", "occupation_title"]]
        .drop_duplicates()
        .sort_values("onetsoc_code")
        .reset_index(drop=True)
    )

    occupation_nodes["occupation_idx"] = occupation_nodes.index

    occupation_nodes = occupation_nodes[
        ["occupation_idx", "onetsoc_code", "occupation_title"]
    ]

    return occupation_nodes

def build_descriptor_nodes(descriptor_rows, descriptor_config):
    descriptor_nodes = (
        descriptor_rows[["descriptor_id", "descriptor_name"]]
        .drop_duplicates()
        .sort_values("descriptor_id")
        .reset_index(drop=True)
    )

    descriptor_nodes["descriptor_idx"] = descriptor_nodes.index

    descriptor_nodes = descriptor_nodes[
        ["descriptor_idx", "descriptor_id", "descriptor_name"]
    ]

    descriptor_nodes = descriptor_nodes.rename(
        columns={
            "descriptor_idx": descriptor_config["idx_col"],
            "descriptor_id": descriptor_config["id_col"],
            "descriptor_name": descriptor_config["name_col"],
        }
    )

    return descriptor_nodes


