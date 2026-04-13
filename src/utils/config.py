"""Configuration helpers for YAML-based experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        TypeError: If the YAML root is not a mapping.
        yaml.YAMLError: If the file contains invalid YAML.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a YAML mapping in '{config_path}', got {type(config).__name__}."
        )

    return config
