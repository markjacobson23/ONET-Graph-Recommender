import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import json

def load_featured_tables(input_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupation_nodes = pd.read_csv(f"{input_dir}/featured_occupation_nodes.csv")
    skill_nodes = pd.read_csv(f"{input_dir}/featured_skill_nodes.csv")
    occupation_skill_edges = pd.read_csv(f"{input_dir}/occupation_skill_edges.csv")
    return occupation_nodes, skill_nodes, occupation_skill_edges

def get_feature_columns(node_table, metadata_columns):
    return node_table.drop(columns=metadata_columns).columns

def build_node_feature_tensors(occupation_nodes, skill_nodes):

    occupation_nodes = (
        occupation_nodes
        .sort_values("occupation_idx")
        .reset_index(drop=True)
        .copy()
    )

    skill_nodes = (
        skill_nodes
        .sort_values("skill_idx")
        .reset_index(drop=True)
        .copy()
    )

    occupation_features = occupation_nodes[
        get_feature_columns(
            occupation_nodes,
            ["onetsoc_code", "occupation_title", "occupation_idx"]
        )
    ]

    skill_features = skill_nodes[
        get_feature_columns(
            skill_nodes,
            ["skill_id", "skill_name", "skill_idx"]
        )
    ]

    occupation_x = torch.tensor(occupation_features.values, dtype=torch.float)
    skill_x = torch.tensor(skill_features.values, dtype=torch.float)

    return occupation_x, skill_x

def build_edge_tensors(occupation_skill_edges):

    # sort the edge table by occupation and skill indices
    occupation_skill_edges = (
        occupation_skill_edges
        .sort_values(["occupation_idx", "skill_idx"])
        .reset_index(drop=True)
        .copy()
    )

    # create edge index
    occupation_requires_skill_edge_index = torch.tensor(
        occupation_skill_edges[["occupation_idx", "skill_idx"]].values,
        dtype=torch.long,
    ).T

    # create edge attributes
    occupation_requires_skill_edge_attr = torch.tensor(
        occupation_skill_edges[["importance", "level"]].values,
        dtype=torch.float,
    )

    return occupation_requires_skill_edge_index, occupation_requires_skill_edge_attr

def build_heterodata(
        occupation_x,
        skill_x,
        occupation_requires_skill_edge_index,
        occupation_requires_skill_edge_attr
    ):

    data = HeteroData()

    skill_rev_requires_occupation_edge_index = torch.flip(occupation_requires_skill_edge_index, dims=[0])

    data["occupation"].x = occupation_x
    data["skill"].x = skill_x
    data["occupation", "requires_skill", "skill"].edge_index = occupation_requires_skill_edge_index
    data["occupation", "requires_skill", "skill"].edge_attr = occupation_requires_skill_edge_attr
    data["skill", "rev_requires_skill", "occupation"].edge_index = skill_rev_requires_occupation_edge_index
    data["skill", "rev_requires_skill", "occupation"].edge_attr = occupation_requires_skill_edge_attr

    return data

def build_metadata(occupation_nodes, skill_nodes, occupation_feature_cols, skill_feature_cols):

    idx_to_onetsoc_code = {
        str(row["occupation_idx"]): row["onetsoc_code"]
        for _, row in occupation_nodes.iterrows()
    }
    idx_to_title = {
        str(row["occupation_idx"]): row["occupation_title"]
        for _, row in occupation_nodes.iterrows()
    }

    idx_to_skill_id = {
        str(row["skill_idx"]): row["skill_id"]
        for _, row in skill_nodes.iterrows()
    }
    idx_to_skill_name = {
        str(row["skill_idx"]): row["skill_name"]
        for _, row in skill_nodes.iterrows()
    }

    metadata = {
        "node_types": {
            "occupation": {
                "num_nodes": len(occupation_nodes),
                "idx_to_onetsoc_code": idx_to_onetsoc_code,
                "idx_to_title": idx_to_title,
                "feature_columns": list(occupation_feature_cols),
            },
            "skill": {
                "num_nodes": len(skill_nodes),
                "idx_to_skill_id": idx_to_skill_id,
                "idx_to_skill_name": idx_to_skill_name,
                "feature_columns": list(skill_feature_cols),
            },
        },
        "edge_types": {
            "occupation__requires_skill__skill": {
                "source_node_type": "occupation",
                "relation": "requires_skill",
                "target_node_type": "skill",
                "edge_attr_columns": ["importance", "level"],
            },
            "skill__rev_requires_skill__occupation": {
                "source_node_type": "skill",
                "relation": "rev_requires_skill",
                "target_node_type": "occupation",
                "edge_attr_columns": ["importance", "level"],
            },
        },
    }
    return metadata

def verify_heterodata(data, occupation_nodes, skill_nodes, occupation_skill_edges):

    assert len(data["occupation"].x) == len(occupation_nodes), (
        "Occupation node count does not match the number of nodes in the HeteroData object"
    )

    assert len(data["skill"].x) == len(skill_nodes), (
        "Skill node count does not match the number of nodes in the HeteroData object"
    )

    assert (
            data["occupation", "requires_skill", "skill"].edge_index.shape
            == torch.Size([2, len(occupation_skill_edges)])
    ), "Edge index shape does not match the number of edges in the HeteroData object"

    assert (
            data["occupation", "requires_skill", "skill"].edge_attr.shape
            == torch.Size([len(occupation_skill_edges), 2])
    ), "Edge attribute shape does not match the number of edges in the HeteroData object"

    assert (
            data["occupation", "requires_skill", "skill"].edge_index[0].max().item()
            < len(occupation_nodes)
    ), "Forward edge_index contains occupation_idx values outside the occupation node range"

    # max skill_idx in edge_index < num_skill_nodes
    assert (
            data["occupation", "requires_skill", "skill"].edge_index[1].max().item()
            < len(skill_nodes)
    ), "Forward edge_index contains skill_idx values outside the skill node range"

    # reverse edge_index shape matches forward edge_index
    assert (
            data["skill", "rev_requires_skill", "occupation"].edge_index.shape
            == data["occupation", "requires_skill", "skill"].edge_index.shape
    ), "Reverse edge_index shape does not match forward edge_index shape"

def save_graph(data, metadata, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.save(data, f"{output_dir}/heterodata.pt")

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

def main():
    input_dir = "../../data/processed/tables/featured"
    output_dir = "../../data/processed/graphs"

    # load tables
    occupation_nodes, skill_nodes, occupation_skill_edges = load_featured_tables(input_dir)

    # build feature tensors
    occupation_x, skill_x = build_node_feature_tensors(occupation_nodes, skill_nodes)

    # build edge tensors
    occupation_requires_skill_edge_index, occupation_requires_skill_edge_attr = build_edge_tensors(occupation_skill_edges)

    # build heterogeneous graph
    data = build_heterodata(occupation_x, skill_x, occupation_requires_skill_edge_index, occupation_requires_skill_edge_attr)

    occupation_feature_cols = get_feature_columns(occupation_nodes, ["onetsoc_code", "occupation_title", "occupation_idx"])
    skill_feature_cols = get_feature_columns(skill_nodes, ["skill_id", "skill_name", "skill_idx"])
    # build metadata
    metadata = build_metadata(occupation_nodes, skill_nodes, occupation_feature_cols, skill_feature_cols )

    # verify the graph
    verify_heterodata(data, occupation_nodes, skill_nodes, occupation_skill_edges)

    # save the graph and metadata
    save_graph(data, metadata, output_dir)

    print("Graph saved successfully.")

data = torch.load(
    "../../data/processed/graphs/heterodata.pt",
    weights_only=False,
)
print(data)
data.validate(raise_on_error=True)
if __name__ == "__main__":
    main()




