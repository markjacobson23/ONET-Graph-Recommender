# O*NET Descriptor ETL

This package contains the config-driven ETL for O*NET occupation-descriptor data.

The active implementation lives in `src/data/onet/descriptors/` and is split into:

- `base/` for raw SQLite loading, node construction, edge construction, and validation
- `featured/` for node feature generation and feature validation
- `configs.py`, `schema.py`, and `io.py` for shared configuration and helpers

The ETL is built around the occupation plus descriptor pattern used by skills, knowledge, and abilities.
