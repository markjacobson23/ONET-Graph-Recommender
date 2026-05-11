import pandas as pd

def build_occupation_nodes(occupation_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the occupation nodes table.
       [occupation_idx, onetsoc_code, occupation_title]
    """

    # drop duplicates and sort by onetsoc_code
    occupation_nodes = (
        occupation_rows[["onetsoc_code", "occupation_title"]]
        .drop_duplicates()
        .sort_values("onetsoc_code")
        .reset_index(drop=True)
    )

    # add occupation_idx column
    occupation_nodes["occupation_idx"] = occupation_nodes.index

    # select the columns needed
    occupation_nodes = occupation_nodes[
        ["occupation_idx", "onetsoc_code", "occupation_title"]
    ]

    return occupation_nodes

def build_descriptor_nodes(descriptor_rows: pd.DataFrame , descriptor_config: dict) -> pd.DataFrame:
    """Build the descriptor nodes table.
       [descriptor_idx, descriptor_id, descriptor_name]
    """

    # drop duplicates and sort by descriptor_id
    descriptor_nodes = (
        descriptor_rows[["descriptor_id", "descriptor_name"]]
        .drop_duplicates()
        .sort_values("descriptor_id")
        .reset_index(drop=True)
    )

    # add descriptor_idx column
    descriptor_nodes["descriptor_idx"] = descriptor_nodes.index

    # select the columns needed
    descriptor_nodes = descriptor_nodes[
        ["descriptor_idx", "descriptor_id", "descriptor_name"]
    ]

    # rename columns according to the descriptor config
    descriptor_nodes = descriptor_nodes.rename(
        columns={
            "descriptor_idx": descriptor_config["idx_col"],
            "descriptor_id": descriptor_config["id_col"],
            "descriptor_name": descriptor_config["name_col"],
        }
    )

    return descriptor_nodes


