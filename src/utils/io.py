"""File system and JSON helpers used across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved directory path as a ``Path`` object.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(paths: list[PathLike]) -> list[Path]:
    """Create multiple directories.

    Args:
        paths: Directory paths to create.

    Returns:
        A list of created or existing directory paths.
    """

    return [ensure_dir(path) for path in paths]


def read_json(path: PathLike) -> dict[str, Any]:
    """Read a JSON file into a dictionary.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        TypeError: If the JSON root is not an object.
    """

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in '{json_path}', got {type(data).__name__}.")

    return data


def write_json(path: PathLike, data: dict[str, Any], indent: int = 2) -> None:
    """Write a dictionary to a JSON file.

    Args:
        path: Output path for the JSON file.
        data: Dictionary to serialize.
        indent: Number of spaces used for pretty printing.
    """

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, sort_keys=True)
        file.write("\n")
