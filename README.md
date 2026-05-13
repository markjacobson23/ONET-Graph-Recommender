# O*NET Graph Recommender

A heterogeneous graph ML pipeline that classifies occupations by SOC code using O*NET skill, knowledge, and ability data. Compares a degree-based majority baseline, MLP, HeteroSAGE, and HeteroTransformer across three graph density variants and five random seeds.

---

## The Problem

The U.S. Department of Labor's O*NET database describes ~900 occupations across hundreds of skill, knowledge, and ability descriptors. The natural structure of this data is a bipartite graph: occupations connect to descriptors with weighted edges representing importance and proficiency levels.

The question this project asks: **can a GNN that propagates information across occupation-descriptor relationships outperform a flat MLP that ignores graph structure when classifying occupations by SOC major group?**

---

## Graph Structure

Three graph variants were built to test how edge density affects model performance:

| Variant | Forward Edges | Description |
|---|---|---|
| `dense` | 107,280 | All occupation-descriptor edges retained |
| `core_broad` | 6,234 | ~6% of edges; importance score ≥ threshold |
| `core_strict` | 1,450 | ~1.4% of edges; high-importance only |

Node types: **OCCUPATION** (894 nodes), **SKILL** (35), **KNOWLEDGE** (33), **ABILITY** (52)

Edge types: `occupation→skill`, `occupation→knowledge`, `occupation→ability` (and reverses)

The graph is heterogeneous — occupations and descriptors are semantically different node types and are handled separately throughout the pipeline using PyTorch Geometric's `HeteroData`.

---

## Results

All models evaluated across 5 random seeds. Metric: test accuracy and macro F1 on held-out occupations.

### Test Accuracy (mean ± std, 5 seeds)

| Model | dense | core_broad | core_strict |
|---|---|---|---|
| Majority class | 0.111 ± 0.017 | 0.111 ± 0.017 | 0.111 ± 0.017 |
| MLP | 0.441 ± 0.024 | 0.428 ± 0.050 | 0.446 ± 0.056 |
| HeteroSAGE | 0.302 ± 0.111 | 0.267 ± 0.058 | 0.439 ± 0.017 |
| **HeteroTransformer** | **0.471 ± 0.039** | 0.391 ± 0.055 | 0.421 ± 0.035 |

### Test Macro F1 (mean ± std, 5 seeds)

| Model | dense | core_broad | core_strict |
|---|---|---|---|
| Majority class | 0.010 | 0.010 | 0.010 |
| MLP | 0.325 ± 0.055 | 0.298 ± 0.060 | 0.315 ± 0.067 |
| HeteroSAGE | 0.172 ± 0.124 | 0.140 ± 0.043 | 0.313 ± 0.045 |
| **HeteroTransformer** | **0.358 ± 0.016** | 0.277 ± 0.067 | 0.310 ± 0.038 |

### Seed wins (best test accuracy per seed)

| Model | dense | core_broad | core_strict |
|---|---|---|---|
| MLP | 2 | 4 | 3 |
| HeteroTransformer | 3 | 1 | 1 |
| HeteroSAGE | 0 | 0 | 1 |

**Key findings:**

- On the `dense` graph, HeteroTransformer is the strongest model (+3.0 pp accuracy, +3.3 pp macro F1 over MLP), and wins 3 of 5 seeds.
- On sparser graphs (`core_broad`, `core_strict`), the MLP matches or beats the GNNs — suggesting graph structure adds signal only when edges are dense enough to carry meaningful neighborhood information.
- HeteroSAGE is high-variance and underperforms on sparse graphs, but becomes competitive on `core_strict` where edges are high-signal by construction.
- The task has 9 SOC major group classes with imbalanced distribution; all models use class-weighted loss to handle this.

---

## Pipeline

```
raw SQLite (O*NET)
  → ETL: base descriptor tables
  → feature engineering: featured tables
  → graph construction: HeteroData (.pt)
  → model training + evaluation: baseline / MLP / GNN
  → results: metrics CSV + comparison figures
```

Run the full pipeline:

```bash
python3 -m src.data.onet.descriptors.base.build_base_tables
python3 -m src.data.onet.descriptors.featured.build_featured_tables
python3 -m src.graph.build_onet_heterodata
python3 -m src.models.evaluation.compare_baselines_vs_gnn
```

---

## Layout

```
configs/           experiment config (YAML)
src/
  core/            config loading
  data/onet/       ETL: raw SQLite → base → featured tables
  graph/           HeteroData construction and label prep
  models/
    baselines/     majority class + MLP
    gnn/           HeteroSAGE and HeteroTransformer
    evaluation/    training loop, metrics, comparison output
docs/              result snapshots and figures
tests/             unit tests
```

---

## Data Source

[O*NET Resource Center](https://www.onetcenter.org/database.html) — public domain occupational data from the U.S. Department of Labor. The raw data is loaded into a local SQLite database and not included in this repo; download instructions are in `docs/data_setup.md`.

---

## Design Notes

**Why heterogeneous GNNs?** Occupations and descriptors are semantically different node types. A homogeneous GNN would treat them identically; `HeteroData` with type-specific message passing keeps the representations separate until the readout layer.

**Why three graph variants?** Edge density is a meaningful hyperparameter when working with bipartite graphs from scored surveys. The O*NET importance scores have a natural cutoff structure, and testing across density levels reveals whether the graph structure is helping (dense) or just adding noise (sparse).

**Why class-weighted loss?** SOC major groups are unbalanced. Without reweighting, models collapse toward predicting the plurality class, inflating accuracy while macro F1 stays near zero.
