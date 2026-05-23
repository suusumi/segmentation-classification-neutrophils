"""Interfaces for future full-smear neutrophil detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CellDetection:
    """Detected cell bounding box in a full blood-smear image."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str = "neutrophil"


class NeutrophilDetector(Protocol):
    """Detection interface for future YOLO integration."""

    name: str

    def detect(self, image_path: Path) -> list[CellDetection]:
        """Return detected neutrophil bounding boxes."""

