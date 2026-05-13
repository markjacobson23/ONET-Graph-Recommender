from __future__ import annotations

import pandas as pd
import torch


def extract_soc_major_group(onetsoc_code: str) -> str:
    """Extract the SOC major group from a full SOC code."""

    return onetsoc_code[:2]


def build_soc_label_mapping(
    occupation_nodes: pd.DataFrame,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build a stable label mapping from SOC major groups."""

    major_groups = (
        occupation_nodes["onetsoc_code"]
        .apply(extract_soc_major_group)
        .sort_values()
        .unique()
        .tolist()
    )

    label_to_idx = {label: idx for idx, label in enumerate(major_groups)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def build_occupation_labels(
    occupation_nodes: pd.DataFrame,
    label_to_idx: dict[str, int],
) -> torch.Tensor:
    """Convert occupation SOC groups into tensor labels."""

    labels = (
        occupation_nodes["onetsoc_code"]
        .apply(extract_soc_major_group)
        .map(label_to_idx)
    )

    return torch.tensor(labels.values, dtype=torch.long)
