# O*NET Skill Graph Recommender

This project builds a graph-based career and skill recommender from O*NET occupation-skill data.

## Current pipeline

1. Load O*NET SQL tables into SQLite.
2. Extract occupation-skill relationships.
3. Build graph-ready base tables:
   - occupation nodes
   - skill nodes
   - occupation-skill edges
4. Add node features derived from occupation-skill edge attributes.
5. Save featured tables for later PyTorch Geometric graph construction.

## Current graph schema

### Node types

- `occupation`
- `skill`

### Edge types

- `occupation -> requires_skill -> skill`

### Edge attributes

- `importance`
- `level`

## Planned next steps

- Convert featured tables into a PyTorch Geometric `HeteroData` object.
- Add reverse edges for message passing.
- Add a baseline recommender.
- Add candidate profile and job posting nodes.
- Train a graph model.