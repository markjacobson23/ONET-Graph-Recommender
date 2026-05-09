import pandas as pd


def load_occupation_nodes(nodes_dir):
    return pd.read_csv(nodes_dir / "occupation_nodes.csv")


def load_descriptor_nodes(nodes_dir, descriptor_config):
    return pd.read_csv(nodes_dir / descriptor_config["node_filename"])


def load_occupation_descriptor_edges(edges_dir, descriptor_config):
    return pd.read_csv(edges_dir / descriptor_config["edge_filename"])