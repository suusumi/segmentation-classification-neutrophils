"""Morphological feature extraction from nucleus masks."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

from src.pipeline.result import MorphologicalFeatures


def extract_morphological_features(mask: np.ndarray) -> MorphologicalFeatures:
    """Extract region properties and count separated nucleus segments."""

    labeled = label(mask.astype(bool))
    regions = regionprops(labeled)
    if not regions:
        return MorphologicalFeatures(
            nucleus_area_px=0,
            nucleus_perimeter_px=0.0,
            nucleus_solidity=0.0,
            nucleus_eccentricity=0.0,
            nucleus_extent=0.0,
            nucleus_bbox_width_px=0,
            nucleus_bbox_height_px=0,
            nucleus_segments=0,
        )

    nucleus = max(regions, key=lambda region: region.area)
    min_row, min_col, max_row, max_col = nucleus.bbox
    return MorphologicalFeatures(
        nucleus_area_px=int(nucleus.area),
        nucleus_perimeter_px=float(nucleus.perimeter),
        nucleus_solidity=float(nucleus.solidity),
        nucleus_eccentricity=float(nucleus.eccentricity),
        nucleus_extent=float(nucleus.extent),
        nucleus_bbox_width_px=int(max_col - min_col),
        nucleus_bbox_height_px=int(max_row - min_row),
        nucleus_segments=int(len(regions)),
    )

