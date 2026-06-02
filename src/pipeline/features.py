"""Извлечение морфологических признаков из масок ядра."""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import label, regionprops

from src.pipeline.result import MorphologicalFeatures
from src.pipeline.segment_counting import SegmentCountResult, count_nucleus_segments


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Возвращает отношение с защитой от деления на ноль."""
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _circularity(area: float, perimeter: float) -> float:
    """Вычисляет 4*pi*area/perimeter^2."""
    if perimeter == 0.0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def empty_morphological_features() -> MorphologicalFeatures:
    """Возвращает вектор признаков с нулевыми значениями."""
    return MorphologicalFeatures(
        nucleus_area_px=0,
        nucleus_perimeter_px=0.0,
        nucleus_circularity=0.0,
        nucleus_solidity=0.0,
        nucleus_eccentricity=0.0,
        nucleus_extent=0.0,
        nucleus_orientation_degrees=0.0,
        nucleus_major_axis_length_px=0.0,
        nucleus_minor_axis_length_px=0.0,
        nucleus_aspect_ratio=0.0,
        nucleus_convex_area_px=0,
        nucleus_filled_area_px=0,
        nucleus_bbox_width_px=0,
        nucleus_bbox_height_px=0,
        nucleus_segments=0,
        segment_area_mean_px=0.0,
        segment_area_min_px=0,
        segment_area_max_px=0,
        mask_foreground_fraction=0.0,
    )


def extract_morphological_features(
    mask: np.ndarray,
    segment_count: SegmentCountResult | None = None,
) -> MorphologicalFeatures:
    """Извлекает свойства области из маски ядра."""
    binary_mask = mask.astype(bool)
    segment_count = segment_count or count_nucleus_segments(binary_mask)
    regions = regionprops(label(binary_mask))
    if not regions:
        return empty_morphological_features()

    nucleus = max(regions, key=lambda region: region.area)
    min_row, min_col, max_row, max_col = nucleus.bbox
    bbox_width = int(max_col - min_col)
    bbox_height = int(max_row - min_row)
    major_axis = float(nucleus.axis_major_length)
    minor_axis = float(nucleus.axis_minor_length)
    foreground_fraction = _safe_ratio(float(binary_mask.sum()), float(binary_mask.size))

    return MorphologicalFeatures(
        nucleus_area_px=int(nucleus.area),
        nucleus_perimeter_px=float(nucleus.perimeter),
        nucleus_circularity=_circularity(float(nucleus.area), float(nucleus.perimeter)),
        nucleus_solidity=float(nucleus.solidity),
        nucleus_eccentricity=float(nucleus.eccentricity),
        nucleus_extent=float(nucleus.extent),
        nucleus_orientation_degrees=float(math.degrees(nucleus.orientation)),
        nucleus_major_axis_length_px=major_axis,
        nucleus_minor_axis_length_px=minor_axis,
        nucleus_aspect_ratio=_safe_ratio(major_axis, minor_axis),
        nucleus_convex_area_px=int(nucleus.area_convex),
        nucleus_filled_area_px=int(nucleus.area_filled),
        nucleus_bbox_width_px=bbox_width,
        nucleus_bbox_height_px=bbox_height,
        nucleus_segments=segment_count.segment_count,
        segment_area_mean_px=segment_count.segment_area_mean_px,
        segment_area_min_px=segment_count.segment_area_min_px,
        segment_area_max_px=segment_count.segment_area_max_px,
        mask_foreground_fraction=foreground_fraction,
    )
