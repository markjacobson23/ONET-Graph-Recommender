# Model Evaluation

This package contains the experiment runner for comparing the baselines and GNN models.

Current files:

- `build_model.py` builds models and optimizers for each experiment
- `train_model.py` contains the shared training loop, evaluation logic, and early stopping
- `compare_baselines_vs_gnn.py` runs the multi-seed comparison and writes the per-seed results CSV
- `src/analysis/soc_classifier/summarize_results.py` builds summary tables from the saved results
- `src/analysis/soc_classifier/make_figures.py` renders the SOC classifier figures
- `src/analysis/soc_classifier/graph_stats.py` computes the graph variant stats CSV

The comparison script loads the saved graph, adds labels and masks, trains the selected models, and writes the flattened per-seed outputs that the analysis scripts consume.
