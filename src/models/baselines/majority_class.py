from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.models.evaluation.train_model import compute_classification_metrics


def majority_class_baseline(
    data: HeteroData,
    include_balanced_accuracy: bool = False,
) -> dict[str, float | int]:
    """Score every occupation with the majority class from the training split."""

    y = data["occupation"].y
    train_mask = data["occupation"].train_mask

    train_labels = y[train_mask]
    majority_class = torch.mode(train_labels).values.item()
    predictions = torch.full_like(y, fill_value=majority_class)

    results: dict[str, float | int] = {}

    for split_name in ["train", "val", "test"]:
        mask = data["occupation"][f"{split_name}_mask"]
        split_metrics = compute_classification_metrics(
            y_true=y[mask],
            y_pred=predictions[mask],
            include_balanced_accuracy=include_balanced_accuracy,
        )
        results[f"{split_name}_accuracy"] = split_metrics["accuracy"]
        results[f"{split_name}_macro_f1"] = split_metrics["macro_f1"]
        if include_balanced_accuracy:
            results[f"{split_name}_balanced_accuracy"] = split_metrics[
                "balanced_accuracy"
            ]

    results["majority_class"] = majority_class
    return results
