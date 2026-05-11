OCCUPATION_NODE_SCHEMA = {
    "idx_col": "occupation_idx",
    "id_col": "onetsoc_code",
    "name_col": "occupation_title",
    "metadata_cols": ["occupation_idx", "onetsoc_code", "occupation_title"],
}


def get_descriptor_node_schema(descriptor_config: dict) -> dict:
    """Get the corresponding node schema for a given descriptor config."""
    return {
        "idx_col": descriptor_config["idx_col"],
        "id_col": descriptor_config["id_col"],
        "name_col": descriptor_config["name_col"],
        "metadata_cols": [
            descriptor_config["idx_col"],
            descriptor_config["id_col"],
            descriptor_config["name_col"],
        ],
    }