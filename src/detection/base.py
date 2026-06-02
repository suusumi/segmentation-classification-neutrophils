"""Интерфейсы для будущего обнаружения нейтрофилов на полном мазке."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CellDetection:
    """Ограничивающая рамка обнаруженной клетки на изображении полного мазка крови."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str = "neutrophil"


class NeutrophilDetector(Protocol):
    """Интерфейс обнаружения для будущей интеграции YOLO."""
    name: str

    def detect(self, image_path: Path) -> list[CellDetection]:
        """Возвращает ограничивающие рамки обнаруженных нейтрофилов."""
