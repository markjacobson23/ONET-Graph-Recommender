from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load the project configuration from YAML."""

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
