from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


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
) -> dict[str, float]:
    """Compute split accuracy for the current model."""

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
        correct = (predictions[mask] == y[mask]).sum().item()
        total = mask.sum().item()
        results[f"{split_name}_accuracy"] = correct / total if total > 0 else 0.0

    return results


def train_with_early_stopping(
    model: torch.nn.Module,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    graph_aware: bool,
    edge_aware: bool,
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
    )
    final_results["best_epoch"] = best_epoch
    final_results["best_val_accuracy"] = best_val_accuracy

    return {
        "model": model,
        "results": final_results,
    }
