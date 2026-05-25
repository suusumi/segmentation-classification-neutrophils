"""Postprocessing for raw nucleus masks."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import clear_border

DEFAULT_BORDER_MARGIN_PX = 8
DEFAULT_CLUSTER_DISTANCE_FRACTION = 0.28
DEFAULT_CLUSTER_MIN_AREA_RATIO = 0.05
DEFAULT_MIN_HOLE_SIZE_PX = 64
DEFAULT_MIN_OBJECT_SIZE_PX = 64


def _keep_primary_nucleus_cluster(
    mask: np.ndarray,
    max_centroid_distance_px: float,
    min_area_ratio: float,
) -> np.ndarray:
    """Keep the main nucleus cluster and drop distant same-color artifacts."""

    labeled = label(mask.astype(bool))
    regions = regionprops(labeled)
    if not regions:
        return mask.astype(bool)

    anchor = max(regions, key=lambda region: region.area)
    min_area_px = max(1.0, float(anchor.area) * min_area_ratio)
    anchor_y, anchor_x = anchor.centroid
    kept_labels = {anchor.label}

    for region in regions:
        if region.label == anchor.label:
            continue

        centroid_y, centroid_x = region.centroid
        distance_px = float(np.hypot(centroid_y - anchor_y, centroid_x - anchor_x))
        if region.area >= min_area_px and distance_px <= max_centroid_distance_px:
            kept_labels.add(region.label)

    return cast(np.ndarray, np.isin(labeled, list(kept_labels)))


def postprocess_mask(
    mask: np.ndarray,
    min_object_size: int = DEFAULT_MIN_OBJECT_SIZE_PX,
    min_hole_size: int = DEFAULT_MIN_HOLE_SIZE_PX,
    border_margin_px: int = DEFAULT_BORDER_MARGIN_PX,
    cluster_distance_fraction: float = DEFAULT_CLUSTER_DISTANCE_FRACTION,
    cluster_min_area_ratio: float = DEFAULT_CLUSTER_MIN_AREA_RATIO,
) -> np.ndarray:
    """Clean a binary nucleus mask."""

    binary = mask.astype(bool)
    cleaned = clear_border(binary, buffer_size=border_margin_px).astype(bool)
    cleaned = remove_small_objects(cleaned, max_size=min_object_size)
    cleaned = remove_small_holes(cleaned, max_size=min_hole_size)
    cleaned = closing(cleaned, footprint=disk(2))
    cleaned = ndi.binary_fill_holes(cleaned)
    max_centroid_distance_px = min(cleaned.shape) * cluster_distance_fraction
    cleaned = _keep_primary_nucleus_cluster(
        cleaned,
        max_centroid_distance_px=max_centroid_distance_px,
        min_area_ratio=cluster_min_area_ratio,
    )
    return cast(np.ndarray, cleaned.astype(bool))
