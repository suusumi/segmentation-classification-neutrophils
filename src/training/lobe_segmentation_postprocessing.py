"""Compatibility imports for lobe segmentation postprocessing.

The implementation belongs to ``src.pipeline`` because it is used by inference.
This module remains so older training scripts or notebooks keep working.
"""

from src.pipeline.lobe_segmentation_postprocessing import (
    LobeSegmentationPrediction,
    postprocess_lobe_segmentation,
)

__all__ = ["LobeSegmentationPrediction", "postprocess_lobe_segmentation"]
