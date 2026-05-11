def get_node_schema(node_config: dict) -> dict:
    """Get the corresponding node schema for a given node config."""
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