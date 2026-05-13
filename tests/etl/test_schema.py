from __future__ import annotations

import pytest

from src.data.onet.descriptors.configs import DESCRIPTOR_CONFIGS, OCCUPATION_CONFIG
from src.data.onet.descriptors.schema import get_node_schema


@pytest.mark.parametrize("node_config", [OCCUPATION_CONFIG, *DESCRIPTOR_CONFIGS.values()])
def test_get_node_schema_returns_expected_fields(node_config: dict) -> None:
    schema = get_node_schema(node_config)

    assert schema["node_type"] == node_config["node_type"]
    assert schema["idx_col"] == node_config["idx_col"]
    assert schema["id_col"] == node_config["id_col"]
    assert schema["name_col"] == node_config["name_col"]
    assert schema["metadata_cols"] == [
        node_config["idx_col"],
        node_config["id_col"],
        node_config["name_col"],
    ]
