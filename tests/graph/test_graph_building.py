from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.onet.descriptors.schema import get_node_schema
from src.graph.build_onet_heterodata import (
    EDGE_ATTR_COLS,
    build_edge_tensor,
    build_node_feature_tensor,
)


def test_build_node_feature_tensor_sorts_by_index() -> None:
    node_table = pd.DataFrame(
        {
            "occupation_idx": [1, 0],
            "onetsoc_code": ["15-0000", "11-0000"],
            "occupation_title": ["B", "A"],
            "f1": [2.0, 4.0],
            "f2": [3.0, 5.0],
        }
    )
    node_schema = get_node_schema(
        {
            "idx_col": "occupation_idx",
            "id_col": "onetsoc_code",
            "name_col": "occupation_title",
            "node_type": "occupation",
        }
    )

    result = build_node_feature_tensor(node_table, node_schema)

    assert result.shape == (2, 2)
    assert result.tolist() == [[4.0, 5.0], [2.0, 3.0]]


def test_build_edge_tensor_sorts_and_shapes() -> None:
    edge_table = pd.DataFrame(
        {
            "occupation_idx": [1, 0, 0],
            "skill_idx": [0, 1, 0],
            "importance": [5.0, 2.0, 4.0],
            "level": [6.0, 3.0, 5.0],
        }
    )
    start_node_schema = {
        "idx_col": "occupation_idx",
        "node_type": "occupation",
    }
    end_node_schema = {
        "idx_col": "skill_idx",
        "node_type": "skill",
    }

    edge_index, edge_attr = build_edge_tensor(
        edge_table=edge_table,
        start_node_schema=start_node_schema,
        end_node_schema=end_node_schema,
        attr_cols=EDGE_ATTR_COLS,
    )

    assert tuple(edge_index.shape) == (2, 3)
    assert tuple(edge_attr.shape) == (3, 2)
    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [[4.0, 5.0], [2.0, 3.0], [5.0, 6.0]]
