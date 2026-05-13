# ETL for O*NET Descriptor Tables

This folder contains the code that turns raw O*NET database tables into graph-ready CSVs.

The active ETL path is:

```text
src/data/onet/descriptors/
```

It handles O*NET tables where occupations connect to descriptor-like concepts such as:

- skills
- knowledge
- abilities

The flow is intentionally split into two stages:

```text
raw SQLite tables
  -> base tables
  -> featured tables
```

## Base tables

Location:

```text
src/data/onet/descriptors/base/
```

The base stage reads raw O*NET data from SQLite and builds the graph identity tables.

Inputs:

```text
data/raw/onet_raw.db
```

Outputs:

```text
data/processed/tables/base/nodes/
data/processed/tables/base/edges/
```

Base tables answer two questions:

```text
What nodes exist?
What edges connect them?
```

## Featured tables

Location:

```text
src/data/onet/descriptors/featured/
```

The featured stage reads the base tables and adds numeric node features for modeling.

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

Featured tables answer one question:

```text
Which numeric attributes should each node carry into the graph?
```

Edges are copied forward because `importance` and `level` are already useful edge attributes.

## Config and schema

Descriptor behavior is controlled by:

```text
src/data/onet/descriptors/configs.py
```

Shared schema and CSV helpers live in:

```text
src/data/onet/descriptors/schema.py
src/data/onet/descriptors/io.py
```

## Current file split

- `base/loader.py` loads raw SQLite rows
- `base/nodes.py` builds node tables
- `base/edges.py` builds indexed edge tables
- `base/verify.py` checks table integrity
- `featured/features.py` builds aggregated node features
- `featured/verify.py` checks featured tables

## Adding a new descriptor

To add another O*NET descriptor table, the usual work is:

1. Add a descriptor config
2. Load the raw table from SQLite
3. Re-run the base and featured builders

The config-driven split keeps the pipeline reusable without copying the whole flow.
