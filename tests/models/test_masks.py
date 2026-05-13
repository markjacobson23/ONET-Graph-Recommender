from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.graph.data import add_train_val_test_masks


def test_add_train_val_test_masks_covers_all_nodes_without_overlap() -> None:
    data = HeteroData()
    data["occupation"].x = torch.zeros((100, 4), dtype=torch.float)
    data["occupation"].num_nodes = 100

    result = add_train_val_test_masks(data, train_ratio=0.7, val_ratio=0.15, seed=42)

    train_mask = result["occupation"].train_mask
    val_mask = result["occupation"].val_mask
    test_mask = result["occupation"].test_mask

    assert train_mask.sum().item() == 70
    assert val_mask.sum().item() == 15
    assert test_mask.sum().item() == 15
    assert torch.logical_and(train_mask, val_mask).sum().item() == 0
    assert torch.logical_and(train_mask, test_mask).sum().item() == 0
    assert torch.logical_and(val_mask, test_mask).sum().item() == 0
    assert torch.logical_or(torch.logical_or(train_mask, val_mask), test_mask).all().item()
