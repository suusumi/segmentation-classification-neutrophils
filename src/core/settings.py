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


def _env_float(name: str, default: float) -> float:
    """Read a float setting from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    """Read an integer setting from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


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
    lobe_boundary_weights_path: Path = field(
        default_factory=lambda: _env_path(
            "LOBE_BOUNDARY_WEIGHTS_PATH",
            project_path("models", "lobe_boundary_unet_mixed.pt"),
        )
    )
    lobe_counter_name: str = field(default_factory=lambda: _env_str("LOBE_COUNTER_NAME", "auto"))
    lobe_foreground_threshold: float = field(
        default_factory=lambda: _env_float("LOBE_FOREGROUND_THRESHOLD", 0.5)
    )
    lobe_boundary_threshold: float = field(
        default_factory=lambda: _env_float("LOBE_BOUNDARY_THRESHOLD", 0.2)
    )
    lobe_min_segment_area_px: int = field(
        default_factory=lambda: _env_int("LOBE_MIN_SEGMENT_AREA_PX", 64)
    )
    yolo_lobe_weights_path: Path = field(
        default_factory=lambda: _env_path(
            "YOLO_LOBE_WEIGHTS_PATH",
            project_path("models", "yolo_lobes_seg.pt"),
        )
    )
    yolo_lobe_confidence: float = field(
        default_factory=lambda: _env_float("YOLO_LOBE_CONFIDENCE", 0.40)
    )
    yolo_lobe_iou: float = field(default_factory=lambda: _env_float("YOLO_LOBE_IOU", 0.30))
    yolo_lobe_image_size: int = field(
        default_factory=lambda: _env_int("YOLO_LOBE_IMAGE_SIZE", 640)
    )
    yolo_lobe_min_mask_area_px: int = field(
        default_factory=lambda: _env_int("YOLO_LOBE_MIN_MASK_AREA_PX", 16)
    )
    yolo_lobe_device: str = field(default_factory=lambda: _env_str("YOLO_LOBE_DEVICE", "auto"))


SETTINGS = AppSettings()
PATHS.ensure_runtime_dirs()
