# O*NET Graph Recommender

This repo turns O*NET occupation and descriptor data into graph-ready tables, builds a PyTorch Geometric `HeteroData` graph, and runs a small set of baseline and GNN experiments on top of it.

## Current shape

The codebase is organized around three main stages:

1. ETL for O*NET occupation-descriptor tables
2. Graph construction and label preparation
3. Baseline and GNN evaluation

Current descriptor coverage:

- skills
- knowledge
- abilities

## Layout

```text
Job_Recommendation_Model/
  configs/
    default.yaml

  data/
    raw/
      onet_raw.db
    processed/
      tables/
        base/
          nodes/
          edges/
        featured/
          nodes/
          edges/
      graphs/

  src/
    core/
      config.py
    data/
      onet/
        descriptors/
          base/
          featured/
          configs.py
          schema.py
          io.py
    graph/
      build_onet_heterodata.py
      data.py
      labels.py
    models/
      baselines/
      evaluation/
      gnn/
```

## Pipeline

```text
raw SQLite data
  -> base tables
  -> featured tables
  -> HeteroData graph
  -> baseline and GNN evaluation
```

## Key commands

Run these from the project root:

```bash
python3 -m src.data.onet.descriptors.base.build_base_tables
python3 -m src.data.onet.descriptors.featured.build_featured_tables
python3 -m src.graph.build_onet_heterodata
python3 -m src.models.evaluation.compare_baselines_vs_gnn
```

## Graph output

The graph builder writes one graph per variant into:

```text
data/processed/graphs/
```

Each graph variant gets a `.pt` graph file and a matching metadata JSON file.

## Results

Current experiment snapshots live in [docs/results.md](docs/results.md).

For O*NET downloads and source data, start at the [O*NET Resource Center](https://www.onetcenter.org/).

The SOC classifier comparison outputs and figures are written under:

```text
data/processed/results/soc_classifier/
```

Highlights from the latest comparison run:

- The majority-class baseline is stable but weak.
- The MLP is the strongest non-graph baseline.
- The HeteroTransformer is the strongest graph model in the current run.

## Notes

- `src/graph/` owns `HeteroData` construction.
- `src/models/baselines/` contains the simple baselines, including the MLP.
- `src/models/gnn/` contains the heterogenous graph models.
- `src/models/evaluation/` handles model construction, training, and comparison output.
