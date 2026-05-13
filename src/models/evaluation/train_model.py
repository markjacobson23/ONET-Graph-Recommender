from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def compute_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    include_balanced_accuracy: bool = False,
) -> dict[str, float]:
    """Compute accuracy, macro F1, and optional balanced accuracy."""

    y_true = y_true.detach().cpu().view(-1)
    y_pred = y_pred.detach().cpu().view(-1)

    total = y_true.numel()
    accuracy = (
        (y_true == y_pred).sum().item() / total if total > 0 else 0.0
    )

    if total == 0 and y_pred.numel() == 0:
        label_values: list[int] = []
    else:
        label_values = torch.unique(torch.cat([y_true, y_pred])).tolist()

    macro_f1_scores: list[float] = []
    balanced_accuracy_recalls: list[float] = []

    for label in label_values:
        true_positive = ((y_true == label) & (y_pred == label)).sum().item()
        false_positive = ((y_true != label) & (y_pred == label)).sum().item()
        false_negative = ((y_true == label) & (y_pred != label)).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator > 0 else 0.0
        )

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        macro_f1_scores.append(f1)

        if include_balanced_accuracy and recall_denominator > 0:
            balanced_accuracy_recalls.append(recall)

    results: dict[str, float] = {
        "accuracy": accuracy,
        "macro_f1": sum(macro_f1_scores) / len(macro_f1_scores)
        if macro_f1_scores
        else 0.0,
    }

    if include_balanced_accuracy:
        results["balanced_accuracy"] = (
            sum(balanced_accuracy_recalls) / len(balanced_accuracy_recalls)
            if balanced_accuracy_recalls
            else 0.0
        )

    return results


def get_model_logits(
    model: torch.nn.Module,
    data: HeteroData,
    graph_aware: bool,
    edge_aware: bool,
) -> torch.Tensor:
    """Run a forward pass with the right argument shape for the model."""

    if graph_aware and edge_aware:
        return model(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
        )

    if graph_aware and not edge_aware:
        return model(
            data.x_dict,
            data.edge_index_dict,
        )

    if not graph_aware and edge_aware:
        raise ValueError("edge_aware=True requires graph_aware=True")

    return model(data["occupation"].x)


def train_one_epoch(
    model: torch.nn.Module,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    graph_aware: bool,
    edge_aware: bool,
) -> float:
    """Train one epoch on the occupation classification task."""

    model.train()
    optimizer.zero_grad()

    logits = get_model_logits(
        model=model,
        data=data,
        graph_aware=graph_aware,
        edge_aware=edge_aware,
    )

    train_mask = data["occupation"].train_mask
    y = data["occupation"].y

    loss = F.cross_entropy(
        logits[train_mask],
        y[train_mask],
    )

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: HeteroData,
    graph_aware: bool,
    edge_aware: bool,
    include_balanced_accuracy: bool = False,
) -> dict[str, float]:
    """Compute split metrics for the current model."""

    model.eval()

    logits = get_model_logits(
        model=model,
        data=data,
        graph_aware=graph_aware,
        edge_aware=edge_aware,
    )

    predictions = logits.argmax(dim=1)
    y = data["occupation"].y

    results: dict[str, float] = {}

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

    return results


def train_with_early_stopping(
    model: torch.nn.Module,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    graph_aware: bool,
    edge_aware: bool,
    include_balanced_accuracy: bool = False,
    num_epochs: int = 2500,
    patience: int = 200,
    print_every: int = 100,
) -> dict[str, object]:
    """Train until validation accuracy stops improving."""

    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
            graph_aware=graph_aware,
            edge_aware=edge_aware,
        )

        results = evaluate(
            model=model,
            data=data,
            graph_aware=graph_aware,
            edge_aware=edge_aware,
            include_balanced_accuracy=include_balanced_accuracy,
        )

        val_accuracy = results["val_accuracy"]

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Train: {results['train_accuracy']:.3f} | "
                f"Val: {results['val_accuracy']:.3f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_results = evaluate(
        model=model,
        data=data,
        graph_aware=graph_aware,
        edge_aware=edge_aware,
        include_balanced_accuracy=include_balanced_accuracy,
    )
    final_results["best_epoch"] = best_epoch
    final_results["best_val_accuracy"] = best_val_accuracy

    return {
        "model": model,
        "results": final_results,
    }
