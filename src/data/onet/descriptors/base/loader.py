from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from src.data.onet.descriptors.configs import ALLOWED_DESCRIPTOR_TABLES


def load_occupation_rows(db_path: Path) -> pd.DataFrame:
    """Load occupation rows from the source SQLite database."""

    query = """
    SELECT
        onetsoc_code,
        title AS occupation_title
    FROM occupation_data;
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def load_descriptor_rows(db_path: Path, source_table: str) -> pd.DataFrame:
    """Load unique descriptor rows for a supported O*NET source table."""

    if source_table not in ALLOWED_DESCRIPTOR_TABLES:
        raise ValueError(f"Invalid source table: {source_table}")

    query = f"""
    SELECT DISTINCT
        d.element_id AS descriptor_id,
        cm.element_name AS descriptor_name
    FROM {source_table} d
    JOIN content_model_reference cm
        ON d.element_id = cm.element_id;
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def load_occupation_descriptor_edge_rows(
    db_path: Path,
    source_table: str,
) -> pd.DataFrame:
    """Load occupation-descriptor edge rows for a supported source table."""

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
        return pd.read_sql_query(query, conn)
