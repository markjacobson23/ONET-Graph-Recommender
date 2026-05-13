# Results

This document captures the current SOC occupation-classification comparison run.

For source data and download guidance, start at the [O*NET Resource Center](https://www.onetcenter.org/).

Run date: 2026-05-13

Command used:

```bash
.venv/bin/python -m src.models.evaluation.compare_baselines_vs_gnn
```

Derived analysis outputs:

- `data/processed/results/soc_classifier/soc_classifier_multiseed_results.csv`
- `data/processed/results/soc_classifier/soc_classifier_multiseed_summary.csv`
- `data/processed/results/soc_classifier/soc_classifier_baseline_improvements.csv`
- `data/processed/results/soc_classifier/soc_classifier_seed_wins.csv`
- `data/processed/results/soc_classifier/graph_variant_stats.csv`

## What Was Compared

- Majority-class baseline
- MLP baseline
- HeteroGraphSAGE
- HeteroTransformer

Each model was run across five seeds on three graph variants:

- `dense`
- `core_broad`
- `core_strict`

## Main Takeaways

- The majority-class baseline stays near 11% test accuracy and serves as the lower bound.
- The best overall graph-model run in this snapshot is `dense` + HeteroTransformer, with mean test accuracy of 0.471 and mean test macro F1 of 0.358.
- The best non-graph baseline is MLP on `core_strict`, with mean test accuracy of 0.446 and mean test macro F1 of 0.315.
- Graph sparsification matters: `core_broad` keeps about 5.81% of dense forward edges, and `core_strict` keeps about 1.35%.

## Summary by Graph Variant

The table below shows the key summary metrics from `soc_classifier_multiseed_summary.csv`.

### Dense

| Model | Test accuracy | Test macro F1 | Best epoch mean |
| --- | ---: | ---: | ---: |
| Majority-class | 0.111 | 0.010 | - |
| MLP | 0.441 | 0.325 | 812.8 |
| HeteroGraphSAGE | 0.247 | 0.172 | 406.8 |
| HeteroTransformer | 0.471 | 0.358 | 578.0 |

### Core broad

| Model | Test accuracy | Test macro F1 | Best epoch mean |
| --- | ---: | ---: | ---: |
| Majority-class | 0.111 | 0.010 | - |
| MLP | 0.428 | 0.298 | 741.2 |
| HeteroGraphSAGE | 0.267 | 0.140 | 446.2 |
| HeteroTransformer | 0.391 | 0.277 | 533.2 |

### Core strict

| Model | Test accuracy | Test macro F1 | Best epoch mean |
| --- | ---: | ---: | ---: |
| Majority-class | 0.111 | 0.010 | - |
| MLP | 0.446 | 0.315 | 639.2 |
| HeteroGraphSAGE | 0.439 | 0.313 | 719.0 |
| HeteroTransformer | 0.423 | 0.299 | 436.2 |

## Baseline Improvements

The baseline-improvement table compares the best model in each graph variant against the majority-class baseline and MLP.

| Graph variant | Best model | Baseline | Accuracy gain | Macro F1 gain |
| --- | --- | --- | ---: | ---: |
| Dense | HeteroTransformer | Majority-class | +36.0 pp | +0.348 |
| Dense | HeteroTransformer | MLP | +3.0 pp | +0.033 |
| Core broad | MLP | Majority-class | +31.7 pp | +0.288 |
| Core broad | MLP | MLP | +0.0 pp | +0.000 |
| Core strict | MLP | Majority-class | +33.5 pp | +0.305 |
| Core strict | MLP | MLP | +0.0 pp | +0.000 |

## Seed Wins

Per-seed winners are counted from the flattened comparison CSV.

| Graph variant | Model | Seed wins |
| --- | --- | ---: |
| Dense | HeteroTransformer | 3 |
| Dense | MLP | 2 |
| Dense | HeteroGraphSAGE | 0 |
| Dense | Majority-class | 0 |
| Core broad | MLP | 4 |
| Core broad | HeteroTransformer | 1 |
| Core broad | HeteroGraphSAGE | 0 |
| Core broad | Majority-class | 0 |
| Core strict | MLP | 3 |
| Core strict | HeteroGraphSAGE | 1 |
| Core strict | HeteroTransformer | 1 |
| Core strict | Majority-class | 0 |

## Graph Statistics

The graph-variant stats table shows how aggressively each graph was sparsified relative to `dense`.

| Graph variant | Forward edges | Forward edges retained | Retention |
| --- | ---: | ---: | ---: |
| Dense | 107,280 | 107,280 | 100.0% |
| Core broad | 6,234 | 6,234 | 5.8% |
| Core strict | 1,450 | 1,450 | 1.4% |

## Notes

- These numbers come from the current comparison output and should be treated as a snapshot, not a permanent benchmark.
- The derived CSVs now live under `data/processed/results/soc_classifier/`.
- The figure set includes `test_accuracy_by_model.png`, `macro_f1_by_model.png`, `graph_variant_edge_counts.png`, `pipeline_flow.png`, and `graph_schema.png`.
