import pandas as pd
from pathlib import Path

from pandas import DataFrame
from pandas.io.parsers import TextFileReader


def load_occupation_nodes(nodes_dir: Path) -> TextFileReader | DataFrame:
    """Load occupation nodes from the CSV file."""
    return pd.read_csv(nodes_dir / "occupation_nodes.csv")


def load_descriptor_nodes(nodes_dir: Path, descriptor_config: dict) -> TextFileReader | DataFrame:
    """Load descriptor nodes from the CSV file."""
    return pd.read_csv(nodes_dir / descriptor_config["node_filename"])


def load_occupation_descriptor_edges(edges_dir: Path, descriptor_config: dict) -> TextFileReader | DataFrame:
    """Load occupation-descriptor edges from the CSV file."""
    return pd.read_csv(edges_dir / descriptor_config["edge_filename"])