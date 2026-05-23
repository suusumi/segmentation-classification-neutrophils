"""Rule-based classification for nucleus segmentation count and shape quality."""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.result import ClassificationResult, MorphologicalFeatures


@dataclass(frozen=True)
class RuleBasedClassifierConfig:
    """Thresholds for interpretable neutrophil classification."""

    hypersegmentation_min_segments: int = 5
    normal_max_segments: int = 4
    low_mask_fraction: float = 0.005
    high_mask_fraction: float = 0.65


def _quality_penalty(
    features: MorphologicalFeatures,
    config: RuleBasedClassifierConfig,
) -> tuple[float, list[str]]:
    """Return confidence penalty and warning reasons for suspicious masks."""

    warnings: list[str] = []
    penalty = 0.0
    if features.mask_foreground_fraction < config.low_mask_fraction:
        warnings.append("very small nucleus mask")
        penalty += 0.3
    if features.mask_foreground_fraction > config.high_mask_fraction:
        warnings.append("very large nucleus mask")
        penalty += 0.3
    if features.nucleus_solidity < 0.45:
        warnings.append("fragmented nucleus mask")
        penalty += 0.15
    return penalty, warnings


def classify_neutrophil(
    features: MorphologicalFeatures,
    config: RuleBasedClassifierConfig | None = None,
) -> ClassificationResult:
    """Classify normal vs hypersegmented neutrophil from nucleus segment count."""

    config = config or RuleBasedClassifierConfig()
    if features.nucleus_segments == 0:
        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            reason="No nucleus mask was detected.",
        )

    penalty, warnings = _quality_penalty(features, config)
    suffix = f" Mask quality warnings: {', '.join(warnings)}." if warnings else ""

    if features.nucleus_segments >= config.hypersegmentation_min_segments:
        return ClassificationResult(
            label="hypersegmentation",
            confidence=max(0.05, 0.82 - penalty),
            reason=(
                "Detected "
                f"{features.nucleus_segments} nucleus segments, meeting the "
                f"hypersegmentation threshold of {config.hypersegmentation_min_segments}."
                f"{suffix}"
            ),
        )
    return ClassificationResult(
        label="normal",
        confidence=max(0.05, 0.72 - penalty),
        reason=(
            "Detected "
            f"{features.nucleus_segments} nucleus segments, below the "
            f"hypersegmentation threshold of {config.hypersegmentation_min_segments}."
            f"{suffix}"
        ),
    )
