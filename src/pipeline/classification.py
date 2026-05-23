"""Rule-based classification for nucleus segmentation count."""

from __future__ import annotations

from src.pipeline.result import ClassificationResult, MorphologicalFeatures


def classify_neutrophil(features: MorphologicalFeatures) -> ClassificationResult:
    """Classify normal vs hypersegmented neutrophil from nucleus segment count."""

    if features.nucleus_segments == 0:
        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            reason="No nucleus mask was detected.",
        )
    if features.nucleus_segments >= 5:
        return ClassificationResult(
            label="hypersegmentation",
            confidence=0.75,
            reason="Detected five or more separated nucleus segments.",
        )
    return ClassificationResult(
        label="normal",
        confidence=0.65,
        reason="Detected fewer than five separated nucleus segments.",
    )

