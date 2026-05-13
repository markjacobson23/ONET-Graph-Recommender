# O*NET Graph Recommender

A heterogeneous graph ML pipeline that classifies occupations by SOC major group using O*NET skill, knowledge, and ability data. Compares a majority-class baseline, a two-layer MLP, HeteroSAGE, and HeteroTransformer across three graph density variants and five random seeds.

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
| `core_broad` | 6,234 | ~6% of edges; importance score ≥ broad threshold |
| `core_strict` | 1,450 | ~1.4% of edges; high-importance only |

Node types: **OCCUPATION** (894), **SKILL** (35), **KNOWLEDGE** (33), **ABILITY** (52)

Edge types: `occupation→skill`, `occupation→knowledge`, `occupation→ability` (and their reverses)

The graph is heterogeneous — occupations and descriptors are semantically different node types handled separately throughout using PyTorch Geometric's `HeteroData`. Edge attributes (importance and proficiency scores) are passed as `edge_attr` in models that support them.

---

## Models

**Majority class baseline** — predicts the most frequent SOC group in the training split for every occupation.

**MLP** — two-layer fully connected network operating on occupation node features only; no graph structure used.

**HeteroSAGE** — two-layer heterogeneous GraphSAGE using `SAGEConv` per edge type, aggregated with `HeteroConv`. Does not use edge attributes.

**HeteroTransformer** — single-layer heterogeneous Transformer using `TransformerConv` per edge type. Uses edge attributes (importance and proficiency scores).

All learned models trained with `cross_entropy` loss, Adam optimizer (lr=0.001, weight_decay=1e-4), early stopping on validation accuracy (patience=200, max 2500 epochs). Best model state is restored before final evaluation.

---

## Results

All models evaluated across 5 random seeds (0–4) with a 70/15/15 train/val/test split.

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

- On the `dense` graph, HeteroTransformer is the strongest model (+3.0 pp accuracy, +3.3 pp macro F1 over MLP) and wins 3 of 5 seeds. The HeteroTransformer uses edge attributes (importance and proficiency scores), which likely explains its advantage on the dense graph where those signals are richest.
- On sparser graphs (`core_broad`, `core_strict`), the MLP matches or beats both GNNs — suggesting graph structure adds meaningful signal only when edge density is high enough.
- HeteroSAGE is the highest-variance model, underperforming badly on sparse graphs but becoming competitive on `core_strict` where remaining edges are high-importance by construction.
- The task has 9 SOC major group classes with unbalanced distribution; macro F1 is the primary metric since it weights all classes equally regardless of frequency.

---

## Pipeline

```
raw SQLite (O*NET)
  → ETL: base descriptor tables
  → feature engineering: featured tables
  → graph construction: HeteroData (.pt)
  → model training + early stopping
  → multi-seed evaluation across graph variants
  → results CSV + summary
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
    baselines/     majority class baseline and MLP
    gnn/           HeteroSAGE and HeteroTransformer
    evaluation/    training loop, metrics, multi-seed comparison
docs/              result snapshots and figures
tests/             unit tests
```

---

## Data Source

[O*NET Resource Center](https://www.onetcenter.org/database.html) — public domain occupational data from the U.S. Department of Labor. The raw data is loaded into a local SQLite database and is not included in this repo. Download the O*NET database files from the link above and follow the setup instructions in `docs/data_setup.md`.

---

## Design Notes

**Why heterogeneous GNNs?** Occupations and descriptors are semantically different node types. A homogeneous GNN would treat them identically; `HeteroData` with type-specific message passing keeps their representations separate until the readout layer, which operates only on occupation nodes.

**Why three graph density variants?** O*NET importance scores have a natural cutoff structure. Testing across density levels reveals whether graph structure helps (dense) or adds noise (sparse) — which turned out to be the most interesting result.

**Why early stopping on validation accuracy?** The model with the best validation accuracy is restored before final evaluation, rather than the model at the final epoch, to avoid overfitting on a task with limited labeled data.
