"""Утилиты конфигурации для настроек экспериментов на базе YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Загружает файл конфигурации YAML в словарь.

    Args:
        path: Путь к файлу конфигурации YAML.

    Returns:
        Словарь со значениями конфигурации.

    Raises:
        FileNotFoundError: Если файл конфигурации не существует.
        TypeError: Если корень YAML не является отображением.
        yaml.YAMLError: Если файл содержит некорректный YAML.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a YAML mapping in '{config_path}', got {type(config).__name__}."
        )

    return config
