import torch
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from torch_geometric.data import HeteroData

from src.graph.labels import build_soc_label_mapping, build_occupation_labels


def load_heterodata(graph_path: Path) -> HeteroData:
    return torch.load(
        graph_path,
        map_location="cpu",
        weights_only=False,
    )

def load_occupation_nodes(featured_nodes_dir: Path) -> TextFileReader | DataFrame:
    return pd.read_csv(featured_nodes_dir / "occupation_nodes.csv")

def add_occupation_labels(
    data: HeteroData,
    occupation_nodes: pd.DataFrame,
) -> tuple[HeteroData, dict, dict]:

    label_to_idx, idx_to_label = build_soc_label_mapping(occupation_nodes)

    labels = build_occupation_labels(
        occupation_nodes,
        label_to_idx,
    )

    data["occupation"].y = labels

    return data, label_to_idx, idx_to_label

def add_train_val_test_masks(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
) -> HeteroData:

    num_nodes = data["occupation"].num_nodes

    generator = torch.Generator()
    generator.manual_seed(seed)

    perm = torch.randperm(num_nodes, generator=generator)

    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)

    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data["occupation"].train_mask = train_mask
    data["occupation"].val_mask = val_mask
    data["occupation"].test_mask = test_mask

    return data

