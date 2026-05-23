"""YOLO detector placeholder.

This adapter keeps the public boundary stable for future full-smear analysis.
Current application flows analyze single-cell images and do not require YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detection.base import CellDetection
from src.services.errors import PipelineError


@dataclass
class YOLONeutrophilDetector:
    """Future YOLO-based neutrophil detector."""

    weights_path: Path
    name: str = "yolo"

    def detect(self, image_path: Path) -> list[CellDetection]:
        """Detect neutrophils on a full smear image."""

        if not self.weights_path.is_file():
            raise PipelineError("YOLO weights are not available yet.")
        raise PipelineError("YOLO inference is not implemented in the first application stage.")

