"""Импорты совместимости для постобработки сегментации долей.

Реализация находится в ``src.pipeline``, потому что используется инференсом.
Этот модуль оставлен, чтобы старые обучающие скрипты и блокноты продолжали работать.
"""
from src.pipeline.lobe_segmentation_postprocessing import (
    LobeSegmentationPrediction,
    postprocess_lobe_segmentation,
)

__all__ = ["LobeSegmentationPrediction", "postprocess_lobe_segmentation"]
