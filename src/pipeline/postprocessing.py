"""Postprocessing for raw nucleus masks."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects


def postprocess_mask(
    mask: np.ndarray,
    min_object_size: int = 64,
    min_hole_size: int = 64,
) -> np.ndarray:
    """Clean a binary nucleus mask."""

    binary = mask.astype(bool)
    cleaned = remove_small_objects(binary, max_size=min_object_size)
    cleaned = remove_small_holes(cleaned, max_size=min_hole_size)
    cleaned = closing(cleaned, footprint=disk(2))
    cleaned = ndi.binary_fill_holes(cleaned)
    return cast(np.ndarray, cleaned.astype(bool))
