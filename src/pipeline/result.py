"""Сериализуемые модели результата анализа."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MorphologicalFeatures:
    """Морфологические дескрипторы, извлеченные из маски ядра."""
    nucleus_area_px: int
    nucleus_perimeter_px: float
    nucleus_circularity: float
    nucleus_solidity: float
    nucleus_eccentricity: float
    nucleus_extent: float
    nucleus_orientation_degrees: float
    nucleus_major_axis_length_px: float
    nucleus_minor_axis_length_px: float
    nucleus_aspect_ratio: float
    nucleus_convex_area_px: int
    nucleus_filled_area_px: int
    nucleus_bbox_width_px: int
    nucleus_bbox_height_px: int
    nucleus_segments: int
    segment_area_mean_px: float
    segment_area_min_px: int
    segment_area_max_px: int
    mask_foreground_fraction: float


@dataclass(frozen=True)
class ClassificationResult:
    """Правиловая классификация сегментации нейтрофилов."""
    label: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Пути к файлам, созданным пайплайном."""
    original_image: str
    mask_image: str
    overlay_image: str
    report_json: str
    report_markdown: str
    log_file: str
    lobe_foreground_image: str | None = None
    lobe_boundary_image: str | None = None
    lobe_components_image: str | None = None
    lobe_overlay_image: str | None = None


@dataclass(frozen=True)
class PipelineMetadata:
    """Детали реализации для воспроизводимости и отладки."""
    pipeline_version: str
    segmenter_name: str
    lobe_counter_name: str
    classifier_name: str
    postprocessing: dict[str, Any]


@dataclass(frozen=True)
class AnalysisResult:
    """Полный результат для одного проанализированного изображения в формате API."""
    analysis_id: str
    status: str
    input_filename: str
    image_width_px: int
    image_height_px: int
    features: MorphologicalFeatures
    classification: ClassificationResult
    artifacts: AnalysisArtifacts
    metadata: PipelineMetadata

    def to_dict(self) -> dict[str, Any]:
        """Преобразует вложенные dataclass-объекты в обычные словари."""
        return asdict(self)
