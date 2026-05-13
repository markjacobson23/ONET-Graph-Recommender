from __future__ import annotations


def get_node_schema(node_config: dict) -> dict:
    """Build the shared node schema used by the ETL and graph code."""

    return {
        "idx_col": node_config["idx_col"],
        "id_col": node_config["id_col"],
        "name_col": node_config["name_col"],
        "node_type": node_config["node_type"],
        "metadata_cols": [
            node_config["idx_col"],
            node_config["id_col"],
            node_config["name_col"],
        ],
    }
