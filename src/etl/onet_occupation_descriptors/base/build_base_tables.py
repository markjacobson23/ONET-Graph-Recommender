from src.utils.config import resolve_project_path, load_config

from src.etl.onet_occupation_descriptors.configs import DESCRIPTOR_CONFIGS
from src.etl.onet_occupation_descriptors.base.loader import (
    load_descriptor_rows,
    load_occupation_rows,
    load_occupation_descriptor_edge_rows,
)
from src.etl.onet_occupation_descriptors.base.nodes import (
    build_occupation_nodes,
    build_descriptor_nodes,
)
from src.etl.onet_occupation_descriptors.base.edges import (
    build_occupation_descriptor_edges,
)
from src.etl.onet_occupation_descriptors.base.verify import (
    verify_descriptor_nodes,
    verify_occupation_nodes,
    verify_occupation_descriptor_edges,
)

from src.etl.onet_occupation_descriptors.io import (
    save_occupation_nodes,
    save_descriptor_nodes,
    save_occupation_descriptor_edges,
)

def main():

    # load config dict
    path_config = load_config()

    # resolve paths
    db_path = resolve_project_path(path_config["paths"]["raw_db_path"])
    base_nodes_dir = resolve_project_path(path_config["paths"]["base_nodes_dir"])
    base_edges_dir = resolve_project_path(path_config["paths"]["base_edges_dir"])

    # build, verify, and save occupation nodes
    occupation_rows = load_occupation_rows(db_path)
    occupation_nodes = build_occupation_nodes(occupation_rows)
    verify_occupation_nodes(occupation_rows, occupation_nodes)
    save_occupation_nodes(occupation_nodes, base_nodes_dir)

    # initialize success string
    success_string = f"Successfully built the following tables:\n"
    success_string += "- occupation_nodes.csv\n"

    # loop through descriptor configs and build, verify, and save node and edge tables
    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():

        print(f"Building {descriptor_name} base-tables...")

        # build, verify, and save descriptor nodes
        descriptor_rows = load_descriptor_rows(db_path, descriptor_config["source_table"])
        descriptor_nodes = build_descriptor_nodes(descriptor_rows, descriptor_config)
        verify_descriptor_nodes(descriptor_rows, descriptor_nodes, descriptor_config)
        save_descriptor_nodes(descriptor_nodes, base_nodes_dir, descriptor_config["node_filename"])

        # build, verify, and save occupation-descriptor edges
        occupation_descriptor_edge_rows = load_occupation_descriptor_edge_rows(
            db_path,
            descriptor_config["source_table"]
        )
        occupation_descriptor_edges = build_occupation_descriptor_edges(
            occupation_descriptor_edge_rows,
            occupation_nodes,
            descriptor_nodes,
            descriptor_config
        )
        verify_occupation_descriptor_edges(
            occupation_descriptor_edge_rows,
            occupation_descriptor_edges,
            descriptor_config
        )
        save_occupation_descriptor_edges(
            occupation_descriptor_edges,
            base_edges_dir,
            descriptor_config["edge_filename"]
        )

        # append successes to success string
        success_string += f"- {descriptor_config['node_filename']}\n"
        success_string += f"- {descriptor_config['edge_filename']}\n"

    print(success_string)

if __name__ == "__main__":
    main()





