
# ETL Pipeline

This folder contains the code that turns raw O*NET data into graph-ready tables.

Right now, the main ETL system is:

```text
src/etl/onet_occupation_descriptors/
````

This handles O*NET data where occupations are connected to descriptor-like things, such as:

* skills
* knowledge areas
* abilities

The goal is to keep this pipeline reusable. If another O*NET table has the same general shape, we should be able to add it with a config instead of writing a whole new pipeline.

## Pipeline layers

The ETL flow has two main stages:

```text
raw SQLite tables
    ↓
base tables
    ↓
featured tables
```

## Base ETL

Location:

```text
src/etl/onet_occupation_descriptors/base/
```

The base ETL reads raw O*NET tables from SQLite and builds the basic graph tables.

Input:

```text
data/raw/onet_raw.db
```

Outputs:

```text
data/processed/tables/base/nodes/
data/processed/tables/base/edges/
```

The base node tables define what nodes exist:

```text
occupation_nodes.csv
skill_nodes.csv
knowledge_nodes.csv
ability_nodes.csv
```

The base edge tables define how occupations connect to those descriptor nodes:

```text
occupation_skill_edges.csv
occupation_knowledge_edges.csv
occupation_ability_edges.csv
```

In short, the base ETL answers:

```text
What are the nodes?
What are the edges?
```

## Featured ETL

Location:

```text
src/etl/onet_occupation_descriptors/featured/
```

The featured ETL reads the base node and edge tables, then adds numeric features to the node tables.

Inputs:

```text
data/processed/tables/base/nodes/
data/processed/tables/base/edges/
```

Outputs:

```text
data/processed/tables/featured/nodes/
data/processed/tables/featured/edges/
```

The featured ETL answers:

```text
What numeric attributes should each node have for modeling?
```

For now, edges are copied forward unchanged because `importance` and `level` are already useful edge attributes.

## Configs

Descriptor behavior is controlled by:

```text
src/etl/onet_occupation_descriptors/configs.py
```

The occupation node has its own config:

```python
OCCUPATION_CONFIG = {
    "node_filename": "occupation_nodes.csv",
    "idx_col": "occupation_idx",
    "id_col": "onetsoc_code",
    "name_col": "occupation_title",
    "node_type": "occupation",
}
```

Each descriptor type also has a config. For example:

```python
"ability": {
    "source_table": "abilities",
    "node_type": "ability",
    "idx_col": "ability_idx",
    "id_col": "ability_id",
    "name_col": "ability_name",
    "node_filename": "ability_nodes.csv",
    "edge_filename": "occupation_ability_edges.csv",
    "relation_name": "requires_ability",
    "feature_prefix": "ability",
    "feature_count_name": "abilities",
}
```

Each descriptor config tells the pipeline:

* which raw SQLite table to read
* what node type to create
* how to name the node index, ID, and name columns
* where to save the node and edge tables
* how to name the PyG relation
* how to name generated feature columns

The config file also defines allowed descriptor tables:

```python
ALLOWED_DESCRIPTOR_TABLES = {
    config["source_table"]
    for config in DESCRIPTOR_CONFIGS.values()
}
```

This keeps dynamic SQL table names restricted to known O*NET tables.

## Schema

Shared node schema logic lives in:

```text
src/etl/onet_occupation_descriptors/schema.py
```

The schema helper turns a node config into a standard schema dictionary:

```python
def get_node_schema(node_config: dict) -> dict:
    return {
        "idx_col": node_config["idx_col"],
        "id_col": node_config["id_col"],
        "name_col": node_config["name_col"],
        "node_type": node_config["node_type"],
        "metadata_cols": [
            node_config["idx_col"],
            node_config["id_col"],
            node_config["name_col"],
        ],
    }
```

For `OCCUPATION_CONFIG`, this means:

```text
idx_col = occupation_idx
id_col = onetsoc_code
name_col = occupation_title
node_type = occupation
metadata_cols = occupation_idx, onetsoc_code, occupation_title
```

For the ability config, this means:

```text
idx_col = ability_idx
id_col = ability_id
name_col = ability_name
node_type = ability
metadata_cols = ability_idx, ability_id, ability_name
```

Metadata columns are not model features. They are used for identity, joins, sorting, and readable labels.

## Shared CSV I/O

Shared CSV loading and saving helpers live in:

```text
src/etl/onet_occupation_descriptors/io.py
```

SQLite query loaders stay in:

```text
src/etl/onet_occupation_descriptors/base/loader.py
```

That keeps raw database access separate from normal file I/O.

## Important indexing rule

Occupation indices are shared across every descriptor edge table.

```text
occupation_idx 112 = the same occupation everywhere
```

Descriptor indices are local to their own node type.

```text
skill_idx 0      = a skill node
knowledge_idx 0  = a knowledge node
ability_idx 0    = an ability node
```

That is okay because `skill`, `knowledge`, and `ability` are different node types in the heterogeneous graph.

## Active occupation universe

The base ETL filters occupation nodes to the occupations that appear in the active descriptor tables.

This avoids creating isolated occupation nodes that have no skill, knowledge, or ability edges.

Right now, the graph uses the intersection of occupations covered by the active descriptor configs.

## Adding a new occupation descriptor

To add a new O*NET descriptor table with the same `importance` / `level` pattern:

1. Import the raw O*NET table into `data/raw/onet_raw.db`.
2. Add a descriptor config to `DESCRIPTOR_CONFIGS`.
3. Run the base ETL.
4. Run the featured ETL.
5. Run the graph builder.

Example:

```bash
python3 -m src.data.onet.base.build_base_tables
python3 -m src.data.onet.featured.build_feature_tables
python3 -m src.graph.build_onet_heterodata
```

## Descriptor-shaped vs. other O*NET data

This pipeline is specifically for O*NET data shaped like:

```text
occupation -> descriptor
```

with edge attributes like:

```text
importance
level
```

Good fits include:

* skills
* knowledge
* abilities
* likely work activities

Other O*NET tables need their own ETL template.

For example, education/training/experience data has a categorical distribution shape:

```text
occupation
element_id
scale_id
category
data_value
```

