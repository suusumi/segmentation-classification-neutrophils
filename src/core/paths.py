"""Утилиты путей относительно проекта.

Приложение не полагается на пользовательские абсолютные пути. Пути времени выполнения
выводятся из корня репозитория и преобразуются обратно в относительные строки для
ответов API и сохраненных метаданных.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Строит путь относительно корня проекта."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_project_dir(*parts: str | Path) -> Path:
    """Создает и возвращает каталог внутри корня проекта."""
    directory = project_path(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_project_relative(path: str | Path) -> Path:
    """Возвращает путь относительно корня проекта, когда это возможно."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved


def to_project_relative_str(path: str | Path) -> str:
    """Возвращает путь в POSIX-стиле для стабильных JSON-ответов на любой ОС."""
    return to_project_relative(path).as_posix()


@dataclass(frozen=True)
class ProjectPaths:
    """Общие каталоги проекта."""

    root: Path = PROJECT_ROOT
    data: Path = project_path("data")
    raw_data: Path = project_path("data", "raw")
    interim_data: Path = project_path("data", "interim")
    processed_data: Path = project_path("data", "processed")
    models: Path = project_path("models")
    outputs: Path = project_path("outputs")
    analyses: Path = project_path("outputs", "analyses")
    reports: Path = project_path("outputs", "reports")
    logs: Path = project_path("outputs", "logs")

    def ensure_runtime_dirs(self) -> None:
        """Создает каталоги, используемые API и пайплайном анализа."""
        for directory in (
            self.raw_data,
            self.interim_data,
            self.processed_data,
            self.models,
            self.outputs,
            self.analyses,
            self.reports,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()
