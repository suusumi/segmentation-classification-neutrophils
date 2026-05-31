"""Postprocessing for lobe foreground and boundary predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops


@dataclass(frozen=True)
class LobeSegmentationPrediction:
    """Postprocessed lobe segmentation result."""

    segment_count: int
    split_mask: np.ndarray
    foreground_mask: np.ndarray
    boundary_mask: np.ndarray


def _filter_labeled_mask(labeled_mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove small labeled regions and relabel the remaining components."""

    filtered = np.zeros(labeled_mask.shape, dtype=np.int32)
    next_label = 1
    for region in regionprops(labeled_mask):
        if region.area < min_area_px:
            continue
        filtered[labeled_mask == region.label] = next_label
        next_label += 1
    return cast(np.ndarray, label(filtered > 0).astype(np.int32))


def _assign_foreground_to_components(
    component_mask: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """Expand component labels across the full predicted foreground."""

    if component_mask.max() == 0:
        return component_mask
    _, nearest_indices = ndi.distance_transform_edt(
        component_mask == 0,
        return_indices=True,
    )
    expanded = component_mask[nearest_indices[0], nearest_indices[1]]
    expanded[~foreground_mask] = 0
    return cast(np.ndarray, expanded.astype(np.int32))


def postprocess_lobe_segmentation(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    foreground_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_segment_area_px: int = 32,
    support_mask: np.ndarray | None = None,
) -> LobeSegmentationPrediction:
    """Convert foreground/boundary probabilities into countable lobe components."""

    foreground_mask = foreground_probability >= foreground_threshold
    boundary_mask = boundary_probability >= boundary_threshold
    if support_mask is not None:
        support = support_mask.astype(bool)
        foreground_mask &= support
        boundary_mask &= support
    split_binary = foreground_mask & ~boundary_mask
    component_cores = _filter_labeled_mask(label(split_binary), min_area_px=min_segment_area_px)
    split_mask = _assign_foreground_to_components(component_cores, foreground_mask)
    return LobeSegmentationPrediction(
        segment_count=int(split_mask.max()),
        split_mask=split_mask,
        foreground_mask=foreground_mask,
        boundary_mask=boundary_mask,
    )
