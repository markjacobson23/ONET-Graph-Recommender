
# O*NET Graph Recommender

This project builds a graph-based career recommendation and occupation analysis pipeline from O*NET data.

The pipeline extracts O*NET occupation-descriptor relationships, converts them into graph-ready node and edge tables, adds aggregate node features, builds a PyTorch Geometric `HeteroData` graph, and provides an explainable baseline recommender from candidate skill profiles.

## Current status

The project currently supports O*NET occupation descriptors:

- skills
- knowledge
- abilities

These descriptor types are handled through a shared config-driven ETL pipeline. Adding another descriptor-shaped O*NET table should mostly require adding a new descriptor config, importing the raw table into SQLite, and rerunning the pipeline.

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
      build_onet_heterodata.py

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
baseline recommender / future graph models
```

## Data layers

### Raw data

Raw O*NET SQL tables are loaded into SQLite:

```text
data/raw/onet_raw.db
```

This base database contains source tables such as:
(see src/etl/readme.md for details on adding new descriptor tables)
* `occupation_data`
* `content_model_reference`
* `scales_reference`
* `skills`
* `knowledge`
* `abilities`
### Base tables

Base tables define graph identities and graph relationships.

```text
data/processed/tables/base/nodes/
  occupation_nodes.csv
  skill_nodes.csv
  knowledge_nodes.csv
  ability_nodes.csv

data/processed/tables/base/edges/
  occupation_skill_edges.csv
  occupation_knowledge_edges.csv
  occupation_ability_edges.csv
```

Node tables define what graph nodes exist.

Edge tables define occupation-descriptor relationships. Current edge attributes are:

* `importance`
* `level`

### Featured tables

Featured tables preserve the base graph structure while adding numeric node features.

```text
data/processed/tables/featured/nodes/
  occupation_nodes.csv
  skill_nodes.csv
  knowledge_nodes.csv
  ability_nodes.csv

data/processed/tables/featured/edges/
  occupation_skill_edges.csv
  occupation_knowledge_edges.csv
  occupation_ability_edges.csv
```

Occupation nodes receive descriptor-summary features such as:

* average skill importance
* number of core skills
* average knowledge level
* number of core knowledge areas
* average ability importance
* number of core abilities

Descriptor nodes receive usage-summary features such as:

* average importance across occupations
* average level across occupations
* number of high-importance occupations
* number of core occupations

## Current graph schema

### Node types

```text
occupation
skill
knowledge
ability
```

### Edge types

```text
occupation -> requires_skill -> skill
skill -> rev_requires_skill -> occupation

occupation -> requires_knowledge -> knowledge
knowledge -> rev_requires_knowledge -> occupation

occupation -> requires_ability -> ability
ability -> rev_requires_ability -> occupation
```

### Edge attributes

```text
importance
level
```

## Configuration

Project paths and runtime settings live in:

```text
configs/default.yaml
```

O*NET descriptor behavior is controlled by:

```text
src/etl/onet_occupation_descriptors/configs.py
```

Each descriptor config defines:

* raw SQLite source table
* node type
* index column
* ID column
* name column
* output node filename
* output edge filename
* graph relation name
* feature naming conventions

## Running the pipeline

Run commands from the project root.

```bash
python3 -m src.data.onet.base.build_base_tables
python3 -m src.data.onet.featured.build_feature_tables
python3 -m src.graph.build_onet_heterodata
python3 -m src.baselines.occupation_skill_overlap
```

## Baseline recommender

The current baseline reads a candidate profile from JSON, matches candidate skills to O*NET skill nodes, scores occupations, and prints an explanation.

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



## Planned next steps

* Stabilize the new graph builder and metadata output.
* Update the baseline recommender to fully use the new `featured/nodes` and `featured/edges` layout.
* Add occupation similarity using raw vectors and learned graph embeddings.
* Train a first GNN model for SOC major-group classification.
* Compare GNN occupation embeddings against simple vector similarity baselines.
* Add more O*NET data families, such as education/training/experience, tasks, technology skills, and related occupations.

````
