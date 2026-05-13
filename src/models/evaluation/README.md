# Model Evaluation

This package contains the experiment runner for comparing the baselines and GNN models.

Current files:

- `build_model.py` builds models and optimizers for each experiment
- `train_model.py` contains the shared training loop, evaluation logic, and early stopping
- `compare_baselines_vs_gnn.py` runs the multi-seed comparison and prints summary metrics

The comparison script loads the saved graph, adds labels and masks, trains the selected models, and reports the resulting accuracies.
