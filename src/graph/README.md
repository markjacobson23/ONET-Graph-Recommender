# Graph Construction

This package turns featured O*NET tables into PyTorch Geometric graph artifacts.

Current modules:

- `build_onet_heterodata.py` builds and saves `HeteroData` plus graph metadata
- `data.py` loads graphs, occupation nodes, labels, and train masks
- `labels.py` builds SOC major-group label mappings and tensors

The graph builder owns `HeteroData` construction. Model code consumes the saved graph output from `data/processed/graphs/`.
