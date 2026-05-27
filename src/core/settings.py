"""Application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.paths import PATHS, project_path


def _env_path(name: str, default: Path) -> Path:
    """Read a path setting from the environment."""

    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else project_path(path)


def _env_str(name: str, default: str) -> str:
    """Read a normalized string setting from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized or default


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for API and pipeline components."""

    app_name: str = "Neutrophil Analysis"
    api_version: str = "0.1.0"
    max_upload_size_mb: int = 20
    allowed_image_extensions: set[str] = field(
        default_factory=lambda: {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    )
    allowed_content_types: set[str] = field(
        default_factory=lambda: {
            "image/bmp",
            "image/jpeg",
            "image/png",
            "image/tiff",
        }
    )
    analysis_db_path: Path = project_path("data", "processed", "analysis", "analysis.sqlite3")
    unet_weights_path: Path = field(
        default_factory=lambda: _env_path(
            "UNET_WEIGHTS_PATH",
            project_path("models", "unet_nucleus.pt"),
        )
    )
    segmenter_name: str = field(default_factory=lambda: _env_str("SEGMENTER_NAME", "auto"))


SETTINGS = AppSettings()
PATHS.ensure_runtime_dirs()

