"""Build the featured node and edge tables used by graph construction."""

from __future__ import annotations

from src.core.config import load_config, resolve_project_path
from src.data.onet.descriptors.configs import DESCRIPTOR_CONFIGS, OCCUPATION_CONFIG
from src.data.onet.descriptors.featured.features import (
    attach_features_to_nodes,
    build_descriptor_features,
    build_occupation_descriptor_features,
    fill_missing_feature_values,
)
from src.data.onet.descriptors.featured.verify import verify_featured_nodes
from src.data.onet.descriptors.io import load_csv_df, save_csv_df
from src.data.onet.descriptors.schema import get_node_schema


def main() -> None:
    """Build and save the featured tables from the base ETL output."""

    path_config = load_config()
    base_nodes_dir = resolve_project_path(path_config["paths"]["base_nodes_dir"])
    base_edges_dir = resolve_project_path(path_config["paths"]["base_edges_dir"])
    featured_nodes_dir = resolve_project_path(path_config["paths"]["featured_nodes_dir"])
    featured_edges_dir = resolve_project_path(path_config["paths"]["featured_edges_dir"])

    base_occupation_nodes = load_csv_df(base_nodes_dir, OCCUPATION_CONFIG["node_filename"])
    featured_occupation_nodes = base_occupation_nodes.copy()

    success_string = "Successfully built the following tables:\n"

    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        print(f"Building {descriptor_name} featured-tables...")

        descriptor_node_schema = get_node_schema(descriptor_config)
        base_descriptor_nodes = load_csv_df(base_nodes_dir, descriptor_config["node_filename"])
        occupation_descriptor_edges = load_csv_df(base_edges_dir, descriptor_config["edge_filename"])

        occupation_features = build_occupation_descriptor_features(
            occupation_descriptor_edges,
            descriptor_config,
        )
        featured_occupation_nodes = attach_features_to_nodes(
            featured_occupation_nodes,
            occupation_features,
            get_node_schema(OCCUPATION_CONFIG)["idx_col"],
        )

        descriptor_features = build_descriptor_features(
            occupation_descriptor_edges,
            descriptor_config,
        )
        featured_descriptor_nodes = attach_features_to_nodes(
            base_descriptor_nodes,
            descriptor_features,
            descriptor_node_schema["idx_col"],
        )
        featured_descriptor_nodes = fill_missing_feature_values(
            featured_descriptor_nodes,
            descriptor_node_schema["metadata_cols"],
        )

        verify_featured_nodes(
            base_descriptor_nodes,
            featured_descriptor_nodes,
            descriptor_node_schema,
        )

        save_csv_df(
            featured_descriptor_nodes,
            featured_nodes_dir,
            descriptor_config["node_filename"],
        )

        # The featured edge table keeps the same structure as the base edge table.
        save_csv_df(
            occupation_descriptor_edges,
            featured_edges_dir,
            descriptor_config["edge_filename"],
        )

        success_string += f"- {descriptor_config['node_filename']}\n"
        success_string += f"- {descriptor_config['edge_filename']}\n"

    featured_occupation_nodes = fill_missing_feature_values(
        featured_occupation_nodes,
        get_node_schema(OCCUPATION_CONFIG)["metadata_cols"],
    )

    verify_featured_nodes(
        base_occupation_nodes,
        featured_occupation_nodes,
        get_node_schema(OCCUPATION_CONFIG),
    )

    save_csv_df(
        featured_occupation_nodes,
        featured_nodes_dir,
        OCCUPATION_CONFIG["node_filename"],
    )

    success_string += f"- {OCCUPATION_CONFIG['node_filename']}\n"
    print(success_string)


if __name__ == "__main__":
    main()
