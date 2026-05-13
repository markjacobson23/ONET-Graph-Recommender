"""Build the base occupation-descriptor tables used by later stages."""

from __future__ import annotations

import pandas as pd

from src.core.config import load_config, resolve_project_path
from src.data.onet.descriptors.base.edges import build_occupation_descriptor_edges
from src.data.onet.descriptors.base.loader import (
    load_descriptor_rows,
    load_occupation_descriptor_edge_rows,
    load_occupation_rows,
)
from src.data.onet.descriptors.base.nodes import (
    build_descriptor_nodes,
    build_occupation_nodes,
)
from src.data.onet.descriptors.base.verify import (
    verify_descriptor_nodes,
    verify_occupation_descriptor_edges,
    verify_occupation_nodes,
)
from src.data.onet.descriptors.configs import DESCRIPTOR_CONFIGS, OCCUPATION_CONFIG
from src.data.onet.descriptors.io import save_csv_df


def main() -> None:
    """Build and save the base node and edge tables."""

    path_config = load_config()
    db_path = resolve_project_path(path_config["paths"]["raw_db_path"])
    base_nodes_dir = resolve_project_path(path_config["paths"]["base_nodes_dir"])
    base_edges_dir = resolve_project_path(path_config["paths"]["base_edges_dir"])

    occupation_descriptor_edge_rows_by_type: dict[str, pd.DataFrame] = {}
    valid_onetsoc_codes: set[str] | None = None

    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        occupation_descriptor_edge_rows = load_occupation_descriptor_edge_rows(
            db_path,
            descriptor_config["source_table"],
        )
        occupation_descriptor_edge_rows_by_type[descriptor_name] = occupation_descriptor_edge_rows

        descriptor_onetsoc_codes = set(
            occupation_descriptor_edge_rows["onetsoc_code"].unique()
        )

        if valid_onetsoc_codes is None:
            valid_onetsoc_codes = descriptor_onetsoc_codes
        else:
            valid_onetsoc_codes &= descriptor_onetsoc_codes

    occupation_rows = load_occupation_rows(db_path)
    occupation_rows = occupation_rows[
        occupation_rows["onetsoc_code"].isin(valid_onetsoc_codes)
    ].copy()

    occupation_nodes = build_occupation_nodes(occupation_rows)
    verify_occupation_nodes(occupation_rows, occupation_nodes)
    save_csv_df(occupation_nodes, base_nodes_dir, OCCUPATION_CONFIG["node_filename"])

    success_string = "Successfully built the following tables:\n"
    success_string += f"- {OCCUPATION_CONFIG['node_filename']}\n"

    for descriptor_name, descriptor_config in DESCRIPTOR_CONFIGS.items():
        print(f"Building {descriptor_name} base-tables...")

        descriptor_rows = load_descriptor_rows(
            db_path,
            descriptor_config["source_table"],
        )

        descriptor_nodes = build_descriptor_nodes(
            descriptor_rows,
            descriptor_config,
        )
        verify_descriptor_nodes(
            descriptor_rows,
            descriptor_nodes,
            descriptor_config,
        )
        save_csv_df(
            descriptor_nodes,
            base_nodes_dir,
            descriptor_config["node_filename"],
        )

        occupation_descriptor_edge_rows = occupation_descriptor_edge_rows_by_type[
            descriptor_name
        ]

        occupation_descriptor_edges = build_occupation_descriptor_edges(
            occupation_descriptor_edge_rows=occupation_descriptor_edge_rows,
            occupation_nodes=occupation_nodes,
            descriptor_nodes=descriptor_nodes,
            descriptor_config=descriptor_config,
        )
        verify_occupation_descriptor_edges(
            occupation_descriptor_edge_rows=occupation_descriptor_edge_rows,
            descriptor_edges=occupation_descriptor_edges,
            descriptor_config=descriptor_config,
        )
        save_csv_df(
            occupation_descriptor_edges,
            base_edges_dir,
            descriptor_config["edge_filename"],
        )

        success_string += f"- {descriptor_config['node_filename']}\n"
        success_string += f"- {descriptor_config['edge_filename']}\n"

    print(success_string)


if __name__ == "__main__":
    main()
