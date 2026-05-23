"""Project-relative path helpers.

The application never relies on user-specific absolute paths. Runtime paths are
derived from the repository root and converted back to relative strings for API
responses and persisted metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Build a path relative to the project root."""

    return PROJECT_ROOT.joinpath(*parts)


def ensure_project_dir(*parts: str | Path) -> Path:
    """Create and return a directory under the project root."""

    directory = project_path(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_project_relative(path: str | Path) -> Path:
    """Return a path relative to the project root when possible."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved


def to_project_relative_str(path: str | Path) -> str:
    """Return a POSIX-style path for stable JSON responses on every OS."""

    return to_project_relative(path).as_posix()


@dataclass(frozen=True)
class ProjectPaths:
    """Common project directories."""

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
        """Create directories used by the API and analysis pipeline."""

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

