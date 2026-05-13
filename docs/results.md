# Results

This document captures the current comparison run for the O*NET occupation classification experiments.

Run date: 2026-05-13

Command used:

```bash
.venv/bin/python -m src.models.evaluation.compare_baselines_vs_gnn
```

## What was compared

- Majority-class baseline
- MLP baseline
- HeteroGraphSAGE
- HeteroTransformer

Each model was run across five seeds on three graph variants:

- `dense`
- `core_broad`
- `core_strict`

## Main takeaways

- The majority-class baseline is stable but weak, with test accuracy around 0.111 to 0.117.
- The MLP baseline is the strongest non-graph baseline, with test accuracy around 0.439 to 0.447.
- The HeteroTransformer is the strongest graph model in this run, with test accuracy ranging from about 0.397 to 0.517 depending on the graph variant.
- HeteroGraphSAGE trails the Transformer here and is more sensitive to the graph variant.

## Summary by graph variant

### Dense

| Model | Train | Val | Test | Best epoch mean |
| --- | ---: | ---: | ---: | ---: |
| Majority-class | 0.122 | 0.118 | 0.111 | - |
| MLP | 0.597 | 0.479 | 0.441 | 812.8 |
| HeteroGraphSAGE | 0.248 | 0.255 | 0.247 | 406.8 |
| HeteroTransformer | 0.693 | 0.512 | 0.517 | 578.0 |

### Core broad

| Model | Train | Val | Test | Best epoch mean |
| --- | ---: | ---: | ---: | ---: |
| Majority-class | 0.122 | 0.118 | 0.111 | - |
| MLP | 0.584 | 0.482 | 0.447 | 754.8 |
| HeteroGraphSAGE | 0.479 | 0.345 | 0.321 | 796.8 |
| HeteroTransformer | 0.743 | 0.436 | 0.397 | 682.8 |

### Core strict

| Model | Train | Val | Test | Best epoch mean |
| --- | ---: | ---: | ---: | ---: |
| Majority-class | 0.122 | 0.118 | 0.111 | - |
| MLP | 0.570 | 0.466 | 0.439 | 639.2 |
| HeteroGraphSAGE | 0.563 | 0.449 | 0.396 | 579.2 |
| HeteroTransformer | 0.634 | 0.470 | 0.422 | 436.2 |

## Notes

- These numbers come from the current comparison script output and should be treated as a snapshot, not a permanent benchmark.
- The dense and core-strict runs matched in this output, which is worth rechecking if the graph filtering rules change.
- If future runs are added, this file should keep only the latest snapshot and a short history of notable changes.
- The derived CSVs now live under `data/processed/results/soc_classifier/`, and the figure set includes a forward-edge retention view in `graph_variant_edge_counts.png` plus separate `pipeline_flow.png` and `graph_schema.png` architecture diagrams for cleaner presentation.
