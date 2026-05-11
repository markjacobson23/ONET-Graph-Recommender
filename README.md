Yep — for now, replace your root `README.md` with this:

````md
# O*NET Graph Recommender

This project builds a graph-based career recommendation pipeline from O*NET occupation data.

The current system extracts O*NET occupation-descriptor relationships, converts them into graph-ready node and edge tables, adds aggregate node features, builds PyTorch Geometric graph artifacts, and provides an explainable baseline recommender from candidate skill profiles.

## Current status

The project currently supports O*NET occupation descriptors such as:

- skills
- knowledge

The pipeline is designed so additional O*NET descriptor types, such as abilities or work activities, can be added with minimal new ETL code by adding a descriptor config.

## Project structure

```text
Job_Recommendation_Model/
  configs/
    default.yaml

  profiles/
    mark_v1.json

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
    etl/
      onet_occupation_descriptors/
        base/
        featured/
        configs.py
        schema.py
        io.py

    graph/

    baselines/
      occupation_skill_overlap.py

    utils/
      config.py
````

## Pipeline overview

```text
O*NET SQLite database
    ↓
base node and edge tables
    ↓
featured node and edge tables
    ↓
PyTorch Geometric HeteroData graph
    ↓
explainable baseline recommender
```

## Data layers

### Raw data

Raw O*NET tables are loaded into SQLite:

```text
data/raw/onet_raw.db
```

The raw database contains O*NET source tables such as:

* `occupation_data`
* `skills`
* `knowledge`
* `content_model_reference`
* `scales_reference`

### Base tables

Base tables contain graph identities and relationship edges.

```text
data/processed/tables/base/nodes/
  occupation_nodes.csv
  skill_nodes.csv
  knowledge_nodes.csv

data/processed/tables/base/edges/
  occupation_skill_edges.csv
  occupation_knowledge_edges.csv
```

Node tables define what graph nodes exist.

Edge tables define relationships between occupations and descriptor nodes. Edge attributes include:

* `importance`
* `level`

### Featured tables

Featured tables preserve the base node and edge structure while adding numeric node features.

```text
data/processed/tables/featured/nodes/
  occupation_nodes.csv
  skill_nodes.csv
  knowledge_nodes.csv

data/processed/tables/featured/edges/
  occupation_skill_edges.csv
  occupation_knowledge_edges.csv
```

Occupation nodes receive descriptor-summary features such as:

* average skill importance
* average skill level
* number of core skills
* average knowledge importance
* average knowledge level
* number of core knowledge areas

Descriptor nodes receive usage-summary features such as:

* average importance across occupations
* average level across occupations
* number of high-importance occupations
* number of core occupations

## Graph schema

Current node types:

```text
occupation
skill
knowledge
```

Current edge types:

```text
occupation -> requires_skill -> skill
skill -> rev_requires_skill -> occupation

occupation -> requires_knowledge -> knowledge
knowledge -> rev_requires_knowledge -> occupation
```

Current edge attributes:

```text
importance
level
```

## Baseline recommender

The baseline recommender reads a candidate profile from JSON, matches candidate skills to O*NET skill nodes, scores occupations, and prints an explanation.

Current scoring rule:

```text
occupation score = sum over matched skills of (importance × level)
```

Example candidate profile:

```json
{
  "candidate_id": "mark_v1",
  "candidate_name": "Mark",
  "skills": [
    "Programming",
    "Mathematics",
    "Critical Thinking",
    "Complex Problem Solving",
    "Systems Analysis"
  ]
}
```

The recommender outputs:

* matched skills
* unmatched skills
* top ranked occupations
* per-skill contribution breakdown for each recommendation

## Configuration

Project paths and runtime settings are stored in:

```text
configs/default.yaml
```

The code resolves paths relative to the project root, so scripts should be run as modules from the root directory.

## Running the pipeline

From the project root:

```bash
python3 -m src.etl.onet_occupation_descriptors.base.build_base_tables
python3 -m src.etl.onet_occupation_descriptors.featured.build_feature_tables
python3 -m src.graph.build_onet_occupation_skill_heterodata
python3 -m src.baselines.occupation_skill_overlap
```

## Current milestone

```text
v0.2 — Descriptor-aware O*NET ETL + explainable skill-overlap recommender
```

Completed:

* SQLite-backed raw O*NET source layer
* descriptor-aware base ETL
* descriptor-aware feature ETL
* YAML path configuration
* PyG `HeteroData` graph builder
* candidate profile input
* explainable baseline occupation ranking

## Planned next steps

* Update the graph builder to fully consume the new `featured/nodes` and `featured/edges` layout.
* Add knowledge-aware recommendation scoring.
* Add more O*NET descriptor types, such as abilities or work activities.
* Add CLI arguments for candidate profile path and `top_k`.
* Add tests for base ETL, feature ETL, and graph construction.
* Build a first graph ML task, likely occupation major-group classification or candidate-to-occupation ranking.

```

This replaces the older skill-only README, which still described PyG graph construction and the baseline recommender as planned rather than already built. :contentReference[oaicite:0]{index=0}
```
