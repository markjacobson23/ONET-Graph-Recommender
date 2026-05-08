import sqlite3
import pandas as pd
from pathlib import Path
from src.utils.config import load_config, resolve_project_path

def load_pivot_occupation_skill_rows(db_path: Path):

    pivot_query = """
    SELECT
        s_im.onetsoc_code,
        o.title AS occupation_title,
        s_im.element_id AS skill_id,
        cm.element_name AS skill_name,
        s_im.data_value AS importance,
        s_lv.data_value AS level
    FROM skills s_im
    JOIN skills s_lv
        ON s_im.onetsoc_code = s_lv.onetsoc_code
       AND s_im.element_id = s_lv.element_id
    JOIN occupation_data o
        ON s_im.onetsoc_code = o.onetsoc_code
    JOIN content_model_reference cm
        ON s_im.element_id = cm.element_id
    WHERE s_im.scale_id = 'IM'
      AND s_lv.scale_id = 'LV';
    
    """

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(pivot_query, conn)

    conn.close()
    return df

def build_node_tables(df: pd.DataFrame):
    # Build unique occupation table
    occupation_nodes = (
        df[["onetsoc_code", "occupation_title"]]
        .drop_duplicates()
        .sort_values("onetsoc_code")
        .reset_index(drop=True)
    )

    # Build unique skill table
    skill_nodes = (
        df[["skill_id", "skill_name"]]
        .drop_duplicates()
        .sort_values("skill_id")
        .reset_index(drop=True)
    )
    return occupation_nodes, skill_nodes

def build_id_maps(occupation_nodes, skill_nodes):
    # mappings
    occupation_id_to_idx = {
        row["onetsoc_code"]: idx
        for idx, row in occupation_nodes.iterrows()
    }

    skill_id_to_idx = {
        row["skill_id"]: idx
        for idx, row in skill_nodes.iterrows()
    }
    return occupation_id_to_idx, skill_id_to_idx

def add_graph_indices(df, occupation_nodes, skill_nodes, occupation_id_to_idx, skill_id_to_idx):
    # add indices to dataframe
    df["occupation_idx"] = df["onetsoc_code"].map(occupation_id_to_idx)
    df["skill_idx"] = df["skill_id"].map(skill_id_to_idx)

    # add node indices to node tables
    skill_nodes["skill_idx"] = skill_nodes["skill_id"].map(skill_id_to_idx)
    skill_nodes = skill_nodes[["skill_idx", "skill_id", "skill_name"]]
    occupation_nodes["occupation_idx"] = occupation_nodes["onetsoc_code"].map(occupation_id_to_idx)
    occupation_nodes = occupation_nodes[["occupation_idx", "onetsoc_code", "occupation_title"]]
    return df, occupation_nodes, skill_nodes

def build_edge_table(df):
    # make edge table
    occupation_skill_edges = df[["occupation_idx", "skill_idx", "importance", "level"]]
    return occupation_skill_edges

def verification_check(occupation_nodes, skill_nodes, occupation_skill_edges, df):
    assert len(occupation_nodes) == df["onetsoc_code"].nunique(), (
        "Occupation node count does not match unique occupations in source data"
    )

    assert len(skill_nodes) == df["skill_id"].nunique(), (
        "Skill node count does not match unique skills in source data"
    )

    assert len(occupation_skill_edges) == len(df), (
        "Edge table row count does not match source dataframe row count"
    )

    assert occupation_skill_edges["occupation_idx"].notna().all(), (
        "Some edges have missing occupation_idx values"
    )

    assert occupation_skill_edges["skill_idx"].notna().all(), (
        "Some edges have missing skill_idx values"
    )

    assert not occupation_skill_edges[["occupation_idx", "skill_idx"]].duplicated().any(), (
        "Duplicate occupation-skill edges found"
    )

    assert occupation_skill_edges["importance"].notna().all(), (
        "Some edges have missing importance values"
    )

    assert occupation_skill_edges["level"].notna().all(), (
        "Some edges have missing level values"
    )

    assert occupation_skill_edges["importance"].between(0, 5).all(), (
        "Importance values outside expected range [0, 5]"
    )

    assert occupation_skill_edges["level"].between(0, 7).all(), (
        "Level values outside expected range [0, 7]"
    )

def save_tables(occupation_nodes, skill_nodes, occupation_skill_edges, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    occupation_nodes.to_csv(output_dir / "occupation_nodes.csv", index=False)
    skill_nodes.to_csv(output_dir / "skill_nodes.csv", index=False)
    occupation_skill_edges.to_csv(output_dir / "occupation_skill_edges.csv", index=False)

def main():
    # load config dict
    config = load_config()

    # resolve paths
    db_path = resolve_project_path(config["paths"]["raw_db_path"])
    output_dir = resolve_project_path(config["paths"]["base_tables_dir"])

    # load and pivot raw data
    raw_df = load_pivot_occupation_skill_rows(db_path)

    # build node tables
    occupation_nodes, skill_nodes = build_node_tables(raw_df)

    # build id maps
    occupation_id_to_idx, skill_id_to_idx = build_id_maps(
        occupation_nodes,
        skill_nodes
    )

    # add indices to dataframe and node tables
    df, occupation_nodes, skill_nodes = add_graph_indices(
        raw_df,
        occupation_nodes,
        skill_nodes,
        occupation_id_to_idx,
        skill_id_to_idx
    )

    # build edge table
    occupation_skill_edges = build_edge_table(df)

    # verification check
    verification_check(occupation_nodes, skill_nodes, occupation_skill_edges, df)

    # save tables
    save_tables(occupation_nodes, skill_nodes, occupation_skill_edges, output_dir)

    print("Tables saved successfully.")
if __name__ == "__main__":
    main()