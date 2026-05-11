import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.etl.onet_occupation_descriptors.configs import (
    DESCRIPTOR_CONFIGS,
    OCCUPATION_CONFIG,
)
from src.etl.onet_occupation_descriptors.schema import get_node_schema
from src.utils.config import load_config, resolve_project_path


EDGE_ATTR_COLS = ["importance", "level"]


def get_feature_columns(node_table: pd.DataFrame, metadata_cols: list[str]) -> list[str]:
    return node_table.drop(columns=metadata_cols).columns


def build_node_metadata(node_table: pd.DataFrame, node_schema: dict) -> dict:
    idx_col = node_schema["idx_col"]
    id_col = node_schema["id_col"]
    name_col = node_schema["name_col"]
    metadata_cols = node_schema["metadata_cols"]

    idx_to_id = {
        str(row[idx_col]): row[id_col]
        for _, row in node_table.iterrows()
    }

    idx_to_name = {
        str(row[idx_col]): row[name_col]
        for _, row in node_table.iterrows()
    }

    feature_columns = get_feature_columns(
        node_table,
        metadata_cols,
    )

    return {
        "node_type": node_schema["node_type"],
        "idx_col": idx_col,
        "id_col": id_col,
        "name_col": name_col,
        "num_nodes": len(node_table),
        "idx_to_id": idx_to_id,
        "idx_to_name": idx_to_name,
        "feature_columns": list(feature_columns),
    }


def build_edge_metadata(
    edge_table: pd.DataFrame,
    start_node_schema: dict,
    end_node_schema: dict,
    relation_name: str,
    edge_attr_cols: list[str],
) -> dict:
    start_node_type = start_node_schema["node_type"]
    end_node_type = end_node_schema["node_type"]

    forward_edge_type_key = f"{start_node_type}__{relation_name}__{end_node_type}"
    reverse_edge_type_key = f"{end_node_type}__rev_{relation_name}__{start_node_type}"

    return {
        forward_edge_type_key: {
            "source_node_type": start_node_type,
            "relation": relation_name,
            "target_node_type": end_node_type,
            "num_edges": len(edge_table),
            "edge_attr_columns": edge_attr_cols,
        },
        reverse_edge_type_key: {
            "source_node_type": end_node_type,
            "relation": f"rev_{relation_name}",
            "target_node_type": start_node_type,
            "num_edges": len(edge_table),
            "edge_attr_columns": edge_attr_cols,
        },
    }


def build_metadata(
    occupation_nodes: pd.DataFrame,
    descriptor_nodes_by_type: dict[str, pd.DataFrame],
    edge_tables_by_type: dict[str, pd.DataFrame],
    edge_attr_cols: list[str],
) -> dict:
    occupation_schema = get_node_schema(OCCUPATION_CONFIG)

    metadata = {
        "node_types": {},
        "edge_types": {},
    }

    metadata["node_types"][OCCUPATION_CONFIG["node_type"]] = build_node_metadata(
        occupation_nodes,
        occupation_schema,
    )

    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        descriptor_schema = get_node_schema(descriptor_config)
        descriptor_nodes = descriptor_nodes_by_type[descriptor_name]
        edge_table = edge_tables_by_type[descriptor_name]

        metadata["node_types"][descriptor_config["node_type"]] = build_node_metadata(
            descriptor_nodes,
            descriptor_schema,
        )

        edge_metadata = build_edge_metadata(
            edge_table=edge_table,
            start_node_schema=occupation_schema,
            end_node_schema=descriptor_schema,
            relation_name=descriptor_config["relation_name"],
            edge_attr_cols=edge_attr_cols,
        )

        metadata["edge_types"].update(edge_metadata)

    return metadata


def build_node_feature_tensor(
    node_table: pd.DataFrame,
    node_schema: dict,
) -> torch.Tensor:
    """Build a node feature tensor from a node table and node schema."""

    idx_col = node_schema["idx_col"]
    metadata_cols = node_schema["metadata_cols"]
    node_type = node_schema["node_type"]

    sorted_nodes = (
        node_table
        .sort_values(idx_col)
        .reset_index(drop=True)
        .copy()
    )

    features = sorted_nodes[
        get_feature_columns(
            sorted_nodes,
            metadata_cols,
        )
    ]

    node_x = torch.tensor(features.values, dtype=torch.float)

    assert len(node_x) == len(node_table), (
        f"{node_type} node count in the node table does not match the number "
        f"of {node_type} nodes in the tensor"
    )

    return node_x


def build_edge_tensor(
    edge_table: pd.DataFrame,
    start_node_schema: dict,
    end_node_schema: dict,
    attr_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge tensors from an edge table and node schemas."""

    start_idx_col = start_node_schema["idx_col"]
    end_idx_col = end_node_schema["idx_col"]
    start_node_type = start_node_schema["node_type"]
    end_node_type = end_node_schema["node_type"]

    edges = (
        edge_table
        .sort_values([start_idx_col, end_idx_col])
        .reset_index(drop=True)
        .copy()
    )

    edge_index = torch.tensor(
        edges[[start_idx_col, end_idx_col]].values,
        dtype=torch.long,
    ).T

    edge_attr = torch.tensor(
        edges[attr_cols].values,
        dtype=torch.float,
    )

    assert tuple(edge_index.shape) == (2, len(edges)), (
        f"{start_node_type}-{end_node_type} index tensor shape does not match "
        f"the shape of {start_node_type}-{end_node_type} edge table"
    )

    assert tuple(edge_attr.shape) == (len(edges), len(attr_cols)), (
        f"{start_node_type}-{end_node_type} attr tensor shape does not match "
        f"the shape of {start_node_type}-{end_node_type} edge table"
    )

    return edge_index, edge_attr


def add_node_to_heterodata(
    x: torch.Tensor,
    node_type: str,
    data: HeteroData,
) -> HeteroData:
    """Add a node type to the heterogeneous graph data."""

    data[node_type].x = x

    return data


def add_edge_to_heterodata(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    data: HeteroData,
    start_node_type: str,
    end_node_type: str,
    end_node_relation_name: str,
) -> HeteroData:
    """Add an edge and its reverse to the heterogeneous graph data."""

    data[
        start_node_type,
        end_node_relation_name,
        end_node_type,
    ].edge_index = edge_index

    data[
        start_node_type,
        end_node_relation_name,
        end_node_type,
    ].edge_attr = edge_attr

    rev_edge_index = torch.flip(edge_index, dims=[0])
    reverse_relation_name = f"rev_{end_node_relation_name}"

    data[
        end_node_type,
        reverse_relation_name,
        start_node_type,
    ].edge_index = rev_edge_index

    data[
        end_node_type,
        reverse_relation_name,
        start_node_type,
    ].edge_attr = edge_attr

    return data


def save_heterodata(data: HeteroData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_dir / "heterodata.pt")


def save_metadata(metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    """Build a heterogeneous graph from the featured occupation and descriptor tables."""

    config = load_config()

    featured_nodes_dir = resolve_project_path(config["paths"]["featured_nodes_dir"])
    featured_edges_dir = resolve_project_path(config["paths"]["featured_edges_dir"])
    processed_graphs_dir = resolve_project_path(config["paths"]["processed_graphs_dir"])

    processed_graphs_dir.mkdir(parents=True, exist_ok=True)

    data = HeteroData()

    descriptor_nodes_by_type = {}
    edge_tables_by_type = {}

    occupation_schema = get_node_schema(OCCUPATION_CONFIG)

    occupation_nodes = pd.read_csv(
        featured_nodes_dir / OCCUPATION_CONFIG["node_filename"]
    )

    occupation_x = build_node_feature_tensor(
        occupation_nodes,
        occupation_schema,
    )

    data = add_node_to_heterodata(
        x=occupation_x,
        node_type=OCCUPATION_CONFIG["node_type"],
        data=data,
    )

    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        print(f"Adding {descriptor_name} to HeteroData...")

        descriptor_schema = get_node_schema(descriptor_config)

        descriptor_nodes = pd.read_csv(
            featured_nodes_dir / descriptor_config["node_filename"]
        )

        descriptor_nodes_by_type[descriptor_name] = descriptor_nodes

        descriptor_x = build_node_feature_tensor(
            descriptor_nodes,
            descriptor_schema,
        )

        data = add_node_to_heterodata(
            x=descriptor_x,
            node_type=descriptor_config["node_type"],
            data=data,
        )

        occupation_descriptor_edges = pd.read_csv(
            featured_edges_dir / descriptor_config["edge_filename"]
        )

        edge_tables_by_type[descriptor_name] = occupation_descriptor_edges

        edge_index, edge_attr = build_edge_tensor(
            edge_table=occupation_descriptor_edges,
            start_node_schema=occupation_schema,
            end_node_schema=descriptor_schema,
            attr_cols=EDGE_ATTR_COLS,
        )

        data = add_edge_to_heterodata(
            edge_index=edge_index,
            edge_attr=edge_attr,
            data=data,
            start_node_type=OCCUPATION_CONFIG["node_type"],
            end_node_type=descriptor_config["node_type"],
            end_node_relation_name=descriptor_config["relation_name"],
        )

    data.validate(raise_on_error=True)

    metadata = build_metadata(
        occupation_nodes=occupation_nodes,
        descriptor_nodes_by_type=descriptor_nodes_by_type,
        edge_tables_by_type=edge_tables_by_type,
        edge_attr_cols=EDGE_ATTR_COLS,
    )

    print(data)

    save_heterodata(data, processed_graphs_dir)
    save_metadata(metadata, processed_graphs_dir)

    print("HeteroData graph and metadata saved successfully.")


if __name__ == "__main__":
    main()




