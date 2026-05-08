import pandas as pd
import sqlite3
from src.etl.onet_occupation_descriptors.configs import ALLOWED_DESCRIPTOR_TABLES

def load_occupation_rows(db_path):
    query = """
    SELECT
        onetsoc_code,
        title AS occupation_title
    FROM occupation_data;
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df

def load_descriptor_rows(db_path, source_table):

    if source_table not in ALLOWED_DESCRIPTOR_TABLES:
        raise ValueError(f"Invalid source table: {source_table}")

    query = f"""
    SELECT DISTINCT
        d.element_id AS descriptor_id,
        cm.element_name AS descriptor_name
    FROM {source_table} d
    JOIN content_model_reference cm
        ON d.element_id = cm.element_id;"""

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def load_occupation_descriptor_edge_rows(db_path, source_table):

    if source_table not in ALLOWED_DESCRIPTOR_TABLES:
        raise ValueError(f"Invalid source table: {source_table}")

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

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


