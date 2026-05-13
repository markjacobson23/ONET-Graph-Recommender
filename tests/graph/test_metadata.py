from __future__ import annotations

import pandas as pd

from src.data.onet.descriptors.configs import DESCRIPTOR_CONFIGS, OCCUPATION_CONFIG
from src.data.onet.descriptors.schema import get_node_schema
from src.graph.build_onet_heterodata import build_edge_metadata, build_metadata, build_node_metadata


def test_build_node_metadata_includes_expected_fields(tiny_occupation_nodes: pd.DataFrame) -> None:
    node_schema = get_node_schema(OCCUPATION_CONFIG)

    metadata = build_node_metadata(tiny_occupation_nodes, node_schema)

    assert metadata["idx_to_id"] == {
        "0": "15-2051.00",
        "1": "17-2141.02",
        "2": "11-0000.00",
    }
    assert metadata["idx_to_name"] == {
        "0": "Data Scientist",
        "1": "Engineer",
        "2": "Manager",
    }
    assert metadata["feature_columns"] == []
    assert metadata["num_nodes"] == 3


def test_build_edge_metadata_includes_reverse_edge() -> None:
    start_node_schema = get_node_schema(OCCUPATION_CONFIG)
    end_node_schema = get_node_schema(DESCRIPTOR_CONFIGS["skill"])
    edge_table = pd.DataFrame(
        {
            "occupation_idx": [0, 1],
            "skill_idx": [0, 1],
            "importance": [4.0, 5.0],
            "level": [5.0, 6.0],
        }
    )

    metadata = build_edge_metadata(
        edge_table=edge_table,
        start_node_schema=start_node_schema,
        end_node_schema=end_node_schema,
        relation_name="requires_skill",
        edge_attr_cols=["importance", "level"],
    )

    assert metadata["occupation__requires_skill__skill"]["relation"] == "requires_skill"
    assert metadata["skill__rev_requires_skill__occupation"]["relation"] == "rev_requires_skill"
    assert metadata["occupation__requires_skill__skill"]["num_edges"] == 2
    assert metadata["skill__rev_requires_skill__occupation"]["edge_attr_columns"] == ["importance", "level"]


def test_build_metadata_includes_all_node_and_edge_types(
    tiny_occupation_nodes: pd.DataFrame,
    tiny_skill_nodes: pd.DataFrame,
    tiny_knowledge_nodes: pd.DataFrame,
    tiny_ability_nodes: pd.DataFrame,
    tiny_occupation_skill_edges: pd.DataFrame,
    tiny_occupation_knowledge_edges: pd.DataFrame,
    tiny_occupation_ability_edges: pd.DataFrame,
) -> None:
    metadata = build_metadata(
        occupation_nodes=tiny_occupation_nodes,
        descriptor_nodes_by_type={
            "skill": tiny_skill_nodes,
            "knowledge": tiny_knowledge_nodes,
            "ability": tiny_ability_nodes,
        },
        edge_tables_by_type={
            "skill": tiny_occupation_skill_edges,
            "knowledge": tiny_occupation_knowledge_edges,
            "ability": tiny_occupation_ability_edges,
        },
        edge_attr_cols=["importance", "level"],
    )

    assert set(metadata["node_types"]) == {"occupation", "skill", "knowledge", "ability"}
    assert "occupation__requires_skill__skill" in metadata["edge_types"]
    assert "skill__rev_requires_skill__occupation" in metadata["edge_types"]
    assert "occupation__requires_knowledge__knowledge" in metadata["edge_types"]
    assert "knowledge__rev_requires_knowledge__occupation" in metadata["edge_types"]
    assert "occupation__requires_ability__ability" in metadata["edge_types"]
    assert "ability__rev_requires_ability__occupation" in metadata["edge_types"]
