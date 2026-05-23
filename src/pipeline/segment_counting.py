"""Nucleus segment counting from binary masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


@dataclass(frozen=True)
class SegmentCountConfig:
    """Parameters controlling segment counting and artifact filtering."""

    min_segment_area_px: int = 64
    min_peak_distance_px: int = 8
    watershed_compactness: float = 0.0


@dataclass(frozen=True)
class SegmentCountResult:
    """Segment-counting output and summary statistics."""

    segment_count: int
    labeled_mask: np.ndarray
    segment_area_mean_px: float
    segment_area_min_px: int
    segment_area_max_px: int


def _empty_result(mask: np.ndarray) -> SegmentCountResult:
    return SegmentCountResult(
        segment_count=0,
        labeled_mask=np.zeros(mask.shape, dtype=np.int32),
        segment_area_mean_px=0.0,
        segment_area_min_px=0,
        segment_area_max_px=0,
    )


def _filter_labeled_segments(labeled_mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove tiny labeled regions and relabel the remaining components."""

    filtered = np.zeros(labeled_mask.shape, dtype=np.int32)
    next_label = 1
    for region in regionprops(labeled_mask):
        if region.area < min_area_px:
            continue
        filtered[labeled_mask == region.label] = next_label
        next_label += 1
    return cast(np.ndarray, label(filtered > 0))


def _watershed_segments(mask: np.ndarray, config: SegmentCountConfig) -> np.ndarray:
    """Split touching nucleus lobes using distance-transform watershed."""

    distance = ndi.distance_transform_edt(mask)
    peak_coordinates = peak_local_max(
        distance,
        labels=mask,
        min_distance=config.min_peak_distance_px,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for marker_id, (row, col) in enumerate(peak_coordinates, start=1):
        markers[row, col] = marker_id

    if markers.max() == 0:
        return cast(np.ndarray, label(mask))

    segmented = watershed(
        -distance,
        markers=markers,
        mask=mask,
        compactness=config.watershed_compactness,
    )
    return cast(np.ndarray, segmented)


def count_nucleus_segments(
    mask: np.ndarray,
    config: SegmentCountConfig | None = None,
) -> SegmentCountResult:
    """Count separated or weakly connected nucleus segments."""

    config = config or SegmentCountConfig()
    binary_mask = mask.astype(bool)
    if not binary_mask.any():
        return _empty_result(binary_mask)

    labeled_mask = _watershed_segments(binary_mask, config)
    labeled_mask = _filter_labeled_segments(labeled_mask, config.min_segment_area_px)
    regions = regionprops(labeled_mask)
    if not regions:
        return _empty_result(binary_mask)

    areas = [int(region.area) for region in regions]
    return SegmentCountResult(
        segment_count=len(areas),
        labeled_mask=labeled_mask,
        segment_area_mean_px=float(np.mean(areas)),
        segment_area_min_px=min(areas),
        segment_area_max_px=max(areas),
    )
