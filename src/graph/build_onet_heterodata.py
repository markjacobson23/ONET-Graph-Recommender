from torch_geometric.data import HeteroData
import json

from src.etl.onet_occupation_descriptors.schema import get_node_schema
from src.utils.config import load_config, resolve_project_path
from pathlib import Path
import pandas as pd
import torch
from src.etl.onet_occupation_descriptors.configs import DESCRIPTOR_CONFIGS, OCCUPATION_CONFIG
def get_feature_columns(node_table: pd.DataFrame, metadata_cols: list[str]) -> list[str]:
    return node_table.drop(columns=metadata_cols).columns

def build_node_feature_tensor(node_table: pd.DataFrame, node_schema: dict) ->  torch.Tensor:
    """Build a node feature tensor from a node table and node schema."""

    # get the relevant metadata columns
    idx_col = node_schema["idx_col"]
    metadata_cols = node_schema["metadata_cols"]
    node_type = node_schema["node_type"]

    # sort the node table by the index column
    sorted_nodes = (
        node_table
        .sort_values(idx_col)
        .reset_index(drop=True)
        .copy()
    )

    # get the feature columns
    features = sorted_nodes[
        get_feature_columns(
            sorted_nodes,
            metadata_cols
        )
    ]

    # convert the feature columns to a tensor
    node_x = torch.tensor(features.values, dtype=torch.float)

    assert len(node_x) == len(node_table), (
        f"{node_type} node count in the node table does not match the number of {node_type} nodes in the Tensor"
    )

    return node_x

def build_edge_tensor(
        edge_table: pd.DataFrame,
        start_node_schema: dict,
        end_node_schema: dict,
        attr_cols: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:

    """Build edge tensors from an edge table and node schemas."""

    start_idx_col = start_node_schema["idx_col"]
    end_idx_col = end_node_schema["idx_col"]
    start_node_type = start_node_schema["node_type"]
    end_node_type = end_node_schema["node_type"]

    # sort the edge table by start and end indices
    edges = (
        edge_table
        .sort_values([start_idx_col, end_idx_col])
        .reset_index(drop=True)
        .copy()
    )

    # create edge index
    edge_index = torch.tensor(
        edges[[start_idx_col, end_idx_col]].values,
        dtype=torch.long,
    ).T

    # create edge attributes
    edge_attr = torch.tensor(
        edges[attr_cols].values,
        dtype=torch.float,
    )

    assert edge_index.shape == (2, len(edges)), (
    f"{start_node_type}-{end_node_type} index Tensor shape does not match the shape of {start_node_type}-{end_node_type} edge table"
    )
    assert edge_attr.shape == (len(edges), len(attr_cols)), (
        f"{start_node_type}-{end_node_type} attr Tensor shape does not match the shape of {start_node_type}-{end_node_type} edge table"
    )

    return edge_index, edge_attr

def add_node_to_heterodata(
        x: torch.Tensor,
        node_type: str,
        data: HeteroData,
    ) -> HeteroData:
    """add a node to the heterogeneous graph data."""

    # add the node to the heterogeneous graph data
    data[node_type].x = x

    return data

def add_edge_to_heterodata(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        data: HeteroData,
        start_node_type: str,
        end_node_type: str,
        end_node_relation_name: str,

):
    """add an edge and its reverse to the heterogeneous graph data."""

    # add the forward edge to the heterogeneous graph data
    data[start_node_type, end_node_relation_name, end_node_type].edge_index = edge_index
    data[start_node_type, end_node_relation_name, end_node_type].edge_attr = edge_attr

    # add the reverse edge to the heterogeneous graph data
    rev_edge_index = torch.flip(edge_index, dims=[0])
    data[end_node_type, f"rev_{end_node_relation_name}", start_node_type].edge_index = rev_edge_index
    data[end_node_type, f"rev_{end_node_relation_name}", start_node_type].edge_attr = edge_attr

    return data

def build_metadata_entry():
    pass

def save_heterodata(data: HeteroData, output_dir: Path):
    torch.save(data, output_dir / "heterodata.pt")

def save_metadata(metadata: dict, output_dir: Path):
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

def main():
    """Build a heterogeneous graph from the featured occupation and descriptor tables."""

    # load config dict and paths
    config = load_config()
    featured_nodes_dir = resolve_project_path(config["paths"]["featured_nodes_dir"])
    featured_edges_dir = resolve_project_path(config["paths"]["featured_edges_dir"])
    processed_graphs_dir = resolve_project_path(config["paths"]["processed_graphs_dir"])

    # safeguard to make sure the output directory exists
    processed_graphs_dir.mkdir(parents=True, exist_ok=True)

    # initialize the HeteroData object
    data = HeteroData()

    # get the occupation node schema for metadata info
    occupation_schema = get_node_schema(OCCUPATION_CONFIG)

    # load the occupation nodes
    occupation_nodes = pd.read_csv(
        featured_nodes_dir / OCCUPATION_CONFIG["node_filename"]
    )

    # build the occupation node feature tensor
    occupation_x = build_node_feature_tensor(
        occupation_nodes,
        occupation_schema,
    )

    # add the occupation tensor to the HeteroData object
    data = add_node_to_heterodata(
        x=occupation_x,
        node_type=OCCUPATION_CONFIG["node_type"],
        data=data,
    )

    # loop through descriptor configs and add node and edge tables to the HeteroData object
    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        print(f"Adding {descriptor_name} to HeteroData...")

        # get the descriptor node schema for metadata info
        descriptor_schema = get_node_schema(descriptor_config)

        # load the descriptor nodes
        descriptor_nodes = pd.read_csv(
            featured_nodes_dir / descriptor_config["node_filename"]
        )

        # build the descriptor node feature tensor
        descriptor_x = build_node_feature_tensor(
            descriptor_nodes,
            descriptor_schema,
        )

        # add the descriptor tensor to the HeteroData object
        data = add_node_to_heterodata(
            x=descriptor_x,
            node_type=descriptor_config["node_type"],
            data=data,
        )

        # load the occupation-descriptor edges
        occupation_descriptor_edges = pd.read_csv(
            featured_edges_dir / descriptor_config["edge_filename"]
        )

        # build the occupation-descriptor edge tensors
        edge_index, edge_attr = build_edge_tensor(
            edge_table=occupation_descriptor_edges,
            start_node_schema=occupation_schema,
            end_node_schema=descriptor_schema,
            attr_cols=["importance", "level"],
        )

        # add the occupation-descriptor edge tensors to the HeteroData object
        data = add_edge_to_heterodata(
            edge_index=edge_index,
            edge_attr=edge_attr,
            data=data,
            start_node_type=OCCUPATION_CONFIG["node_type"],
            end_node_type=descriptor_config["node_type"],
            end_node_relation_name=descriptor_config["relation_name"],
        )

    # validate the HeteroData object
    data.validate(raise_on_error=True)

    print(data)

    # save the HeteroData object
    save_heterodata(data, processed_graphs_dir)

    print("HeteroData graph saved successfully.")

if __name__ == "__main__":
    main()




