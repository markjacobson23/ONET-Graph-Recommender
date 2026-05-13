import pandas as pd
from pathlib import Path

def build_occupation_descriptor_edges(
    occupation_descriptor_edge_rows,
    occupation_nodes,
    descriptor_nodes,
    descriptor_config,
)-> pd.DataFrame:
    """Build the occupation-descriptor edges table.

        This function builds the occupation-descriptor edges table by merging
        the occupation node metadata, descriptor node metadata,
        and the raw edge rows loaded from the source data.

        Args:
            occupation_descriptor_edge_rows: the raw edge rows loaded from the source data.
                    [onetsoc_code, descriptor_id, importance, level]

            occupation_nodes: occupation node metadata.
                    [occupation_idx, onetsoc_code, occupation_title]

            descriptor_nodes: descriptor node metadata.
                    [descriptor_idx, descriptor_id, descriptor_name]

            descriptor_config: descriptor configuration parameters.
                    [see src/data/onet/configs.py for details]

        Returns:
            occupation_descriptor_edges: the refined occupation-descriptor edges table.
                    [occupation_idx, descriptor_idx, importance, level]
        """

    # get the id and index column names from the descriptor config
    id_col = descriptor_config["id_col"]
    idx_col = descriptor_config["idx_col"]

    # merge the edge table with the occupation node metadata
    occupation_descriptor_edges = occupation_descriptor_edge_rows.merge(
        occupation_nodes[["onetsoc_code", "occupation_idx"]],
        on="onetsoc_code",
        how="left"
    )

    # merge the edge table with the descriptor node metadata
    occupation_descriptor_edges = occupation_descriptor_edges.merge(
        descriptor_nodes[[id_col, idx_col]],
        left_on="descriptor_id",
        right_on=id_col,
        how="left"
    )

    # refine the edge table to include only the necessary columns
    occupation_descriptor_edges = occupation_descriptor_edges[
        ["occupation_idx", idx_col, "importance", "level"]
    ]

    # sort the edge table by occupation index and descriptor index
    occupation_descriptor_edges = (
        occupation_descriptor_edges
        .sort_values(by=["occupation_idx", idx_col])
        .reset_index(drop=True)
    )

    return occupation_descriptor_edges
