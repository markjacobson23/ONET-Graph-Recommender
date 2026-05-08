
def verify_occupation_nodes(
    occupation_rows,
    occupation_nodes,
):
    assert len(occupation_nodes) == occupation_rows["onetsoc_code"].nunique(), (
        "Occupation node count does not match unique occupations in source data"
    )

def verify_descriptor_nodes(
    descriptor_rows,
    descriptor_nodes,
    descriptor_config,
):
    descriptor_name = descriptor_config["node_type"]

    assert len(descriptor_nodes) == descriptor_rows["descriptor_id"].nunique(), (
        f"{descriptor_name} node count does not match unique {descriptor_name}s in source data"
    )

def verify_occupation_descriptor_edges(
    occupation_descriptor_edge_rows,
    descriptor_edges,
    descriptor_config
):

    descriptor_name = descriptor_config["node_type"]
    id_col = descriptor_config["id_col"]
    idx_col = descriptor_config["idx_col"]

    assert len(descriptor_edges) == len(occupation_descriptor_edge_rows), (
        "Edge table row count does not match source dataframe row count"
    )

    assert descriptor_edges["occupation_idx"].notna().all(), (
        "Some edges have missing occupation_idx values"
    )

    assert descriptor_edges[idx_col].notna().all(), (
        f"Some edges have missing {idx_col} values"
    )

    assert not descriptor_edges[["occupation_idx", idx_col]].duplicated().any(), (
        f"Duplicate occupation-{descriptor_name} edges found"
    )

    assert descriptor_edges["importance"].notna().all(), (
        "Some edges have missing importance values"
    )

    assert descriptor_edges["level"].notna().all(), (
        "Some edges have missing level values"
    )

    assert descriptor_edges["importance"].between(0, 5).all(), (
        "Importance values outside expected range [0, 5]"
    )

    assert descriptor_edges["level"].between(0, 7).all(), (
        "Level values outside expected range [0, 7]"
    )



