import pandas as pd
import torch


def extract_soc_major_group(onetsoc_code: str) -> str:
    return onetsoc_code[:2]

def build_soc_label_mapping(occupation_nodes: pd.DataFrame)-> tuple[dict, dict]:
    major_groups = (
        occupation_nodes["onetsoc_code"]
        .apply(extract_soc_major_group)
        .sort_values()
        .unique()
        .tolist()
    )

    label_to_idx = {
        label: idx
        for idx, label in enumerate(major_groups)
    }

    idx_to_label = {
        idx: label
        for label, idx in label_to_idx.items()
    }

    return label_to_idx, idx_to_label

def build_occupation_labels(occupation_nodes, label_to_idx):
    labels = (
        occupation_nodes["onetsoc_code"]
        .apply(extract_soc_major_group)
        .map(label_to_idx)
    )

    return torch.tensor(labels.values, dtype=torch.long)
