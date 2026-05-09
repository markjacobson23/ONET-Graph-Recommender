from src.etl.onet_occupation_descriptors.configs import DESCRIPTOR_CONFIGS
from src.etl.onet_occupation_descriptors.schema import (
    OCCUPATION_NODE_SCHEMA,
    get_descriptor_node_schema,
)
from src.etl.onet_occupation_descriptors.io import (
    save_occupation_nodes,
    save_descriptor_nodes,
    save_occupation_descriptor_edges,
)
from src.etl.onet_occupation_descriptors.featured.features import (
    attach_features_to_nodes,
    build_descriptor_features,
    build_occupation_descriptor_features,
    fill_missing_feature_values,
)
from src.etl.onet_occupation_descriptors.featured.loader import (
    load_descriptor_nodes,
    load_occupation_descriptor_edges,
    load_occupation_nodes,
)
from src.etl.onet_occupation_descriptors.featured.verify import (
    verify_featured_nodes,
)
from src.utils.config import load_config, resolve_project_path


def main():
    # load config dict
    path_config = load_config()

    # resolve paths
    base_nodes_dir = resolve_project_path(path_config["paths"]["base_nodes_dir"])
    base_edges_dir = resolve_project_path(path_config["paths"]["base_edges_dir"])
    featured_nodes_dir = resolve_project_path(path_config["paths"]["featured_nodes_dir"])
    featured_edges_dir = resolve_project_path(path_config["paths"]["featured_edges_dir"])

    # load and copy base occupation nodes
    base_occupation_nodes = load_occupation_nodes(base_nodes_dir)
    featured_occupation_nodes = base_occupation_nodes.copy()

    # initialize success string
    success_string = "Successfully built the following tables:\n"

    # loop through descriptor configs and build, verify, and save node and edge tables
    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        print(f"Building {descriptor_name} featured-tables...")

        # get descriptor node schema
        descriptor_node_schema = get_descriptor_node_schema(descriptor_config)

        # load descriptor nodes
        base_descriptor_nodes = load_descriptor_nodes(
            base_nodes_dir,
            descriptor_config,
        )

        # load occupation-descriptor edges
        occupation_descriptor_edges = load_occupation_descriptor_edges(
            base_edges_dir,
            descriptor_config,
        )

        # build occupation features
        occupation_features = build_occupation_descriptor_features(
            occupation_descriptor_edges,
            descriptor_config,
        )

        # attach occupation features to occupation nodes
        featured_occupation_nodes = attach_features_to_nodes(
            featured_occupation_nodes,
            occupation_features,
            OCCUPATION_NODE_SCHEMA["idx_col"],
        )

        # build descriptor features
        descriptor_features = build_descriptor_features(
            occupation_descriptor_edges,
            descriptor_config,
        )

        # attach descriptor features to descriptor nodes
        featured_descriptor_nodes = attach_features_to_nodes(
            base_descriptor_nodes,
            descriptor_features,
            descriptor_node_schema["idx_col"],
        )

        # fill missing feature values with 0
        featured_descriptor_nodes = fill_missing_feature_values(
            featured_descriptor_nodes,
            descriptor_node_schema["metadata_cols"],
        )

        # verify that the descriptor node table is as expected
        verify_featured_nodes(
            base_descriptor_nodes,
            featured_descriptor_nodes,
            descriptor_node_schema,
        )

        # save the descriptor node table
        save_descriptor_nodes(
            featured_descriptor_nodes,
            featured_nodes_dir,
            descriptor_config["node_filename"],
        )

        # save the occupation-descriptor edge table
        # (same as base edge table so no extra verify)
        save_occupation_descriptor_edges(
            occupation_descriptor_edges,
            featured_edges_dir,
            descriptor_config["edge_filename"],
        )

        # append successes to success string
        success_string += f"- {descriptor_config['node_filename']}\n"
        success_string += f"- {descriptor_config['edge_filename']}\n"

    # fill missing occupation feature values with 0
    featured_occupation_nodes = fill_missing_feature_values(
        featured_occupation_nodes,
        OCCUPATION_NODE_SCHEMA["metadata_cols"],
    )

    # verify that the occupation node table is as expected
    verify_featured_nodes(
        base_occupation_nodes,
        featured_occupation_nodes,
        OCCUPATION_NODE_SCHEMA,
    )

    # save the occupation node table
    save_occupation_nodes(
        featured_occupation_nodes,
        featured_nodes_dir,
    )

    success_string += "- occupation_nodes.csv\n"
    print(success_string)


if __name__ == "__main__":
    main()





