from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def occupation_config() -> dict:
    return {
        "node_filename": "occupation_nodes.csv",
        "idx_col": "occupation_idx",
        "id_col": "onetsoc_code",
        "name_col": "occupation_title",
        "node_type": "occupation",
    }


@pytest.fixture
def skill_config() -> dict:
    return {
        "source_table": "skills",
        "node_type": "skill",
        "idx_col": "skill_idx",
        "id_col": "skill_id",
        "name_col": "skill_name",
        "node_filename": "skill_nodes.csv",
        "edge_filename": "occupation_skill_edges.csv",
        "relation_name": "requires_skill",
        "feature_prefix": "skill",
        "feature_count_name": "skills",
    }


@pytest.fixture
def ability_config() -> dict:
    return {
        "source_table": "abilities",
        "node_type": "ability",
        "idx_col": "ability_idx",
        "id_col": "ability_id",
        "name_col": "ability_name",
        "node_filename": "ability_nodes.csv",
        "edge_filename": "occupation_ability_edges.csv",
        "relation_name": "requires_ability",
        "feature_prefix": "ability",
        "feature_count_name": "abilities",
    }


@pytest.fixture
def knowledge_config() -> dict:
    return {
        "source_table": "knowledge",
        "node_type": "knowledge",
        "idx_col": "knowledge_idx",
        "id_col": "knowledge_id",
        "name_col": "knowledge_name",
        "node_filename": "knowledge_nodes.csv",
        "edge_filename": "occupation_knowledge_edges.csv",
        "relation_name": "requires_knowledge",
        "feature_prefix": "knowledge",
        "feature_count_name": "knowledge",
    }


@pytest.fixture
def descriptor_configs(skill_config, knowledge_config, ability_config):
    return {
        "skill": skill_config,
        "knowledge": knowledge_config,
        "ability": ability_config,
    }


@pytest.fixture
def tiny_occupation_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [0, 1, 2],
            "onetsoc_code": ["15-2051.00", "17-2141.02", "11-0000.00"],
            "occupation_title": ["Data Scientist", "Engineer", "Manager"],
        }
    )


@pytest.fixture
def tiny_skill_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "skill_idx": [0, 1],
            "skill_id": ["skill_a", "skill_b"],
            "skill_name": ["Skill A", "Skill B"],
        }
    )


@pytest.fixture
def tiny_ability_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ability_idx": [0, 1],
            "ability_id": ["ability_a", "ability_b"],
            "ability_name": ["Ability A", "Ability B"],
        }
    )


@pytest.fixture
def tiny_knowledge_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "knowledge_idx": [0, 1],
            "knowledge_id": ["knowledge_a", "knowledge_b"],
            "knowledge_name": ["Knowledge A", "Knowledge B"],
        }
    )


@pytest.fixture
def tiny_occupation_skill_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [0, 0, 1],
            "skill_idx": [0, 1, 0],
            "importance": [4.0, 2.0, 5.0],
            "level": [5.0, 3.0, 6.0],
        }
    )


@pytest.fixture
def tiny_occupation_ability_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [0, 1],
            "ability_idx": [0, 1],
            "importance": [4.5, 3.5],
            "level": [5.5, 4.5],
        }
    )


@pytest.fixture
def tiny_occupation_knowledge_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [1, 2],
            "knowledge_idx": [0, 1],
            "importance": [4.0, 5.0],
            "level": [5.0, 6.0],
        }
    )


@pytest.fixture
def tiny_descriptor_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [0, 0, 1],
            "skill_idx": [0, 1, 0],
            "importance": [4.0, 2.0, 5.0],
            "level": [5.0, 3.0, 6.0],
        }
    )


@pytest.fixture
def tiny_node_table_with_missing_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occupation_idx": [0, 1],
            "onetsoc_code": ["15-2051.00", "17-2141.02"],
            "occupation_title": ["Data Scientist", "Engineer"],
            "avg_skill_importance": [4.5, None],
            "avg_skill_level": [5.0, None],
        }
    )


@pytest.fixture
def tiny_heterodata() -> HeteroData:
    data = HeteroData()
    data["occupation"].x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float,
    )
    data["skill"].x = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        dtype=torch.float,
    )

    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[4.0, 5.0], [2.0, 3.0], [5.0, 6.0]],
        dtype=torch.float,
    )

    data["occupation", "requires_skill", "skill"].edge_index = edge_index
    data["occupation", "requires_skill", "skill"].edge_attr = edge_attr
    data["skill", "rev_requires_skill", "occupation"].edge_index = torch.flip(
        edge_index,
        dims=[0],
    )
    data["skill", "rev_requires_skill", "occupation"].edge_attr = edge_attr

    data["occupation"].num_nodes = 3
    data["skill"].num_nodes = 2

    return data
