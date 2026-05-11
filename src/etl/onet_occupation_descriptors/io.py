import pandas as pd
from pathlib import Path


def save_occupation_descriptor_edges(occupation_descriptor_edges: pd.DataFrame , output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    occupation_descriptor_edges.to_csv(output_dir / filename, index=False)

def save_occupation_nodes(occupation_nodes: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    occupation_nodes.to_csv(output_dir / "occupation_nodes.csv", index=False)

def save_descriptor_nodes(descriptor_nodes: pd.DataFrame, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_nodes.to_csv(output_dir / filename, index=False)


