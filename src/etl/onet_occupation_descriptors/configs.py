
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
        "feature_prefix": "skill",
        "feature_count_name": "skills",
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
        "feature_prefix": "knowledge",
        "feature_count_name": "knowledge"
    },
    "ability": {
        "source_table": "abilities",
        "node_type": "ability",
        "idx_col": "ability_idx",
        "id_col": "ability_id",
        "name_col": "ability_name",
        "node_filename": "ability_nodes.csv",
        "edge_filename": "occupation_ability_edges.csv",
        "relation_name": "requires_ability",
        "feature_prefix": "ability",
        "feature_count_name": "abilities",
    },
}

ALLOWED_DESCRIPTOR_TABLES = {
    config["source_table"]
    for config in DESCRIPTOR_CONFIGS.values()
}