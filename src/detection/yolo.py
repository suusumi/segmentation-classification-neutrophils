"""Заготовка детектора YOLO.

Этот адаптер сохраняет стабильную публичную границу для будущего анализа полного мазка.
Текущие сценарии приложения анализируют изображения отдельных клеток и не требуют YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detection.base import CellDetection
from src.services.errors import PipelineError


@dataclass
class YOLONeutrophilDetector:
    """Будущий детектор нейтрофилов на базе YOLO."""
    weights_path: Path
    name: str = "yolo"

    def detect(self, image_path: Path) -> list[CellDetection]:
        """Обнаруживает нейтрофилы на изображении полного мазка."""
        if not self.weights_path.is_file():
            raise PipelineError("YOLO weights are not available yet.")
        raise PipelineError("YOLO inference is not implemented in the first application stage.")
