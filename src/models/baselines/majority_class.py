import torch


def majority_class_baseline(data):
    y = data["occupation"].y
    train_mask = data["occupation"].train_mask

    train_labels = y[train_mask]

    majority_class = torch.mode(train_labels).values.item()

    predictions = torch.full_like(y, fill_value=majority_class)

    results = {}

    for split_name in ["train", "val", "test"]:
        mask = data["occupation"][f"{split_name}_mask"]

        correct = (predictions[mask] == y[mask]).sum().item()
        total = mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        results[f"{split_name}_accuracy"] = accuracy

    results["majority_class"] = majority_class

    return results

