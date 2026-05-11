import pandas as pd
import sqlite3
from src.etl.onet_occupation_descriptors.configs import ALLOWED_DESCRIPTOR_TABLES
from pathlib import Path
def load_occupation_rows(db_path: Path) -> pd.DataFrame:
    """Load occupation data rows from the SQLite database."""

    # select onetsoc_code and occupation_title from the occupation_data table
    query = """
    SELECT
        onetsoc_code,
        title AS occupation_title
    FROM occupation_data;
    """

    # load the data into a pandas DataFrame
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df

def load_descriptor_rows(db_path: Path, source_table: str) -> pd.DataFrame:
    """load descriptor data rows from the SQLite database."""

    # check if the source table is allowed
    if source_table not in ALLOWED_DESCRIPTOR_TABLES:
        raise ValueError(f"Invalid source table: {source_table}")

    # select distinct descriptor_id and descriptor_name from the source table
    query = f"""
    SELECT DISTINCT
        d.element_id AS descriptor_id,
        cm.element_name AS descriptor_name
    FROM {source_table} d
    JOIN content_model_reference cm
        ON d.element_id = cm.element_id;"""

    # load the data into a pandas DataFrame
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def load_occupation_descriptor_edge_rows(db_path: Path, source_table: str):
    """Load occupation-descriptor edge rows from the SQLite database."""

    # check if the source table is allowed
    if source_table not in ALLOWED_DESCRIPTOR_TABLES:
        raise ValueError(f"Invalid source table: {source_table}")

    # select occupation_code, descriptor_id, importance, level from the source tables
    query = f"""
    SELECT
        d_im.onetsoc_code,
        d_im.element_id AS descriptor_id,
        d_im.data_value AS importance,
        d_lv.data_value AS level
    FROM {source_table} d_im
    JOIN {source_table} d_lv
        ON d_im.onetsoc_code = d_lv.onetsoc_code
       AND d_im.element_id = d_lv.element_id
    WHERE d_im.scale_id = 'IM'
      AND d_lv.scale_id = 'LV';
    """

    # load the data into a pandas DataFrame
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


