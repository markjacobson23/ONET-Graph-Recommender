from __future__ import annotations

from src.data.onet.descriptors.configs import ALLOWED_DESCRIPTOR_TABLES, DESCRIPTOR_CONFIGS


def test_descriptor_configs_have_required_keys() -> None:
    required_keys = {
        "source_table",
        "node_type",
        "idx_col",
        "id_col",
        "name_col",
        "node_filename",
        "edge_filename",
        "relation_name",
        "feature_prefix",
        "feature_count_name",
    }

    for config in DESCRIPTOR_CONFIGS.values():
        assert required_keys.issubset(config.keys())


def test_allowed_descriptor_tables_match_descriptor_configs() -> None:
    assert ALLOWED_DESCRIPTOR_TABLES == {
        config["source_table"]
        for config in DESCRIPTOR_CONFIGS.values()
    }
