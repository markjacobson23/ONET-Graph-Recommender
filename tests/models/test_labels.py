from __future__ import annotations

import pandas as pd
import torch

from src.graph.labels import build_occupation_labels, build_soc_label_mapping, extract_soc_major_group


def test_extract_soc_major_group() -> None:
    assert extract_soc_major_group("15-2051.00") == "15"
    assert extract_soc_major_group("17-2141.02") == "17"


def test_build_soc_label_mapping_is_contiguous(tiny_occupation_nodes: pd.DataFrame) -> None:
    label_to_idx, idx_to_label = build_soc_label_mapping(tiny_occupation_nodes)

    assert label_to_idx == {"11": 0, "15": 1, "17": 2}
    assert idx_to_label == {0: "11", 1: "15", 2: "17"}


def test_build_occupation_labels_returns_long_tensor(tiny_occupation_nodes: pd.DataFrame) -> None:
    label_to_idx, _ = build_soc_label_mapping(tiny_occupation_nodes)
    labels = build_occupation_labels(tiny_occupation_nodes, label_to_idx)

    assert isinstance(labels, torch.Tensor)
    assert labels.dtype == torch.long
    assert len(labels) == len(tiny_occupation_nodes)
    assert labels.tolist() == [1, 2, 0]
