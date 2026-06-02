"""Утилиты файловой системы и JSON, используемые во всем проекте."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Создает каталог, если он еще не существует.

    Args:
        path: Путь каталога, который нужно создать.

    Returns:
        Разрешенный путь каталога как объект ``Path``.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(paths: list[PathLike]) -> list[Path]:
    """Создает несколько каталогов.

    Args:
        paths: Пути каталогов, которые нужно создать.

    Returns:
        Список путей созданных или уже существующих каталогов.
    """
    return [ensure_dir(path) for path in paths]


def read_json(path: PathLike) -> dict[str, Any]:
    """Читает JSON-файл в словарь.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Разобранное содержимое JSON.

    Raises:
        FileNotFoundError: Если файл не существует.
        json.JSONDecodeError: Если файл содержит некорректный JSON.
        TypeError: Если корень JSON не является объектом.
    """
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in '{json_path}', got {type(data).__name__}.")

    return data


def write_json(path: PathLike, data: dict[str, Any], indent: int = 2) -> None:
    """Записывает словарь в JSON-файл.

    Args:
        path: Выходной путь для JSON-файла.
        data: Словарь для сериализации.
        indent: Количество пробелов для форматированного вывода.
    """
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, sort_keys=True)
        file.write("\n")
