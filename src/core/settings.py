"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.paths import PATHS, project_path


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
    unet_weights_path: Path = project_path("models", "unet_nucleus.pt")
    segmenter_name: str = "threshold"


SETTINGS = AppSettings()
PATHS.ensure_runtime_dirs()

