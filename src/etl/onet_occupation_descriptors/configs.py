DESCRIPTOR_CONFIGS = {
    "skill": {
        "source_table": "skills",
        "node_type": "skill",
        "idx_col": "skill_idx",
        "id_col": "skill_id",
        "name_col": "skill_name",
        "node_filename": "skill_nodes.csv",
        "edge_filename": "occupation_skill_edges.csv",
        "relation_name": "requires_skill",
    },
    "knowledge": {
        "source_table": "knowledge",
        "node_type": "knowledge",
        "idx_col": "knowledge_idx",
        "id_col": "knowledge_id",
        "name_col": "knowledge_name",
        "node_filename": "knowledge_nodes.csv",
        "edge_filename": "occupation_knowledge_edges.csv",
        "relation_name": "requires_knowledge",
    },
}

ALLOWED_DESCRIPTOR_TABLES = {
    config["source_table"]
    for config in DESCRIPTOR_CONFIGS.values()
}