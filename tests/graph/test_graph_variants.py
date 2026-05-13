from __future__ import annotations

import pandas as pd
import pytest

from src.graph.build_onet_heterodata import filter_edge_table


def test_filter_edge_table_variants() -> None:
    edge_table = pd.DataFrame(
        {
            "importance": [3.0, 4.0, 3.0, 4.0],
            "level": [4.0, 4.0, 5.0, 5.0],
        }
    )

    dense = filter_edge_table(edge_table, "dense")
    core_broad = filter_edge_table(edge_table, "core_broad")
    core_strict = filter_edge_table(edge_table, "core_strict")

    assert len(dense) == 4
    assert len(core_broad) == 3
    assert len(core_strict) == 1


def test_filter_edge_table_unknown_variant_raises() -> None:
    edge_table = pd.DataFrame(
        {
            "importance": [4.0],
            "level": [5.0],
        }
    )

    with pytest.raises(ValueError, match="Unknown graph variant"):
        filter_edge_table(edge_table, "unknown")
