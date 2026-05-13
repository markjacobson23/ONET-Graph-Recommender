from __future__ import annotations

import pandas as pd

from src.data.onet.descriptors.base.verify import (
    verify_descriptor_nodes,
    verify_occupation_descriptor_edges,
    verify_occupation_nodes,
)
from src.data.onet.descriptors.featured.verify import verify_featured_nodes


def test_verify_occupation_nodes_passes(tiny_occupation_nodes: pd.DataFrame) -> None:
    occupation_rows = tiny_occupation_nodes[["onetsoc_code", "occupation_title"]].copy()

    verify_occupation_nodes(occupation_rows, tiny_occupation_nodes)


def test_verify_descriptor_nodes_passes(tiny_skill_nodes: pd.DataFrame) -> None:
    descriptor_rows = pd.DataFrame(
        {
            "descriptor_id": ["skill_a", "skill_b"],
            "descriptor_name": ["Skill A", "Skill B"],
        }
    )
    descriptor_config = {
        "node_type": "skill",
        "idx_col": "skill_idx",
    }

    verify_descriptor_nodes(descriptor_rows, tiny_skill_nodes, descriptor_config)


def test_verify_occupation_descriptor_edges_passes(
    tiny_occupation_nodes: pd.DataFrame,
    tiny_skill_nodes: pd.DataFrame,
    tiny_occupation_skill_edges: pd.DataFrame,
) -> None:
    occupation_descriptor_edge_rows = pd.DataFrame(
        {
            "onetsoc_code": ["15-2051.00", "15-2051.00", "17-2141.02"],
            "descriptor_id": ["skill_a", "skill_b", "skill_a"],
            "importance": [4.0, 2.0, 5.0],
            "level": [5.0, 3.0, 6.0],
        }
    )
    descriptor_config = {
        "node_type": "skill",
        "idx_col": "skill_idx",
    }

    verify_occupation_descriptor_edges(
        occupation_descriptor_edge_rows=occupation_descriptor_edge_rows,
        descriptor_edges=tiny_occupation_skill_edges,
        descriptor_config=descriptor_config,
    )


def test_verify_featured_nodes_passes(
    tiny_occupation_nodes: pd.DataFrame,
) -> None:
    featured_nodes = tiny_occupation_nodes.copy()
    featured_nodes["avg_skill_importance"] = [4.5, 0.0, 1.0]
    featured_nodes["avg_skill_level"] = [5.0, 0.0, 1.0]

    node_schema = {
        "idx_col": "occupation_idx",
        "metadata_cols": ["occupation_idx", "onetsoc_code", "occupation_title"],
    }

    verify_featured_nodes(tiny_occupation_nodes, featured_nodes, node_schema)
