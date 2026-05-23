"""Artifact writers for masks, overlays, and reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.core.paths import to_project_relative_str
from src.pipeline.result import AnalysisResult
from src.utils.io import write_json


def save_mask(mask: np.ndarray, path: Path) -> Path:
    """Save a binary mask as an 8-bit PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image.save(path)
    return path


def save_overlay(rgb_image: np.ndarray, mask: np.ndarray, path: Path) -> Path:
    """Save a transparent red mask overlay on the original RGB image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray(rgb_image.astype(np.uint8)).convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask.astype(bool)] = [220, 30, 30, 110]
    overlay_image = Image.fromarray(overlay)
    Image.alpha_composite(base, overlay_image).save(path)
    return path


def save_report_json(result: AnalysisResult, path: Path) -> Path:
    """Persist the complete result as JSON."""

    write_json(path, result.to_dict())
    return path


def save_report_markdown(result: AnalysisResult, path: Path) -> Path:
    """Persist a compact human-readable report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"# Analysis {result.analysis_id}",
            "",
            f"Status: {result.status}",
            f"Input: {result.input_filename}",
            f"Image size: {result.image_width_px}x{result.image_height_px}px",
            f"Classification: {result.classification.label}",
            f"Confidence: {result.classification.confidence:.2f}",
            f"Reason: {result.classification.reason}",
            "",
            "## Features",
            f"- Nucleus area: {result.features.nucleus_area_px}px",
            f"- Nucleus perimeter: {result.features.nucleus_perimeter_px:.2f}px",
            f"- Nucleus circularity: {result.features.nucleus_circularity:.3f}",
            f"- Nucleus solidity: {result.features.nucleus_solidity:.3f}",
            f"- Nucleus eccentricity: {result.features.nucleus_eccentricity:.3f}",
            f"- Nucleus extent: {result.features.nucleus_extent:.3f}",
            f"- Nucleus aspect ratio: {result.features.nucleus_aspect_ratio:.3f}",
            f"- Foreground fraction: {result.features.mask_foreground_fraction:.3f}",
            f"- Nucleus segments: {result.features.nucleus_segments}",
            f"- Segment area mean: {result.features.segment_area_mean_px:.2f}px",
            "",
            "## Pipeline",
            f"- Version: {result.metadata.pipeline_version}",
            f"- Segmenter: {result.metadata.segmenter_name}",
            f"- Classifier: {result.metadata.classifier_name}",
            "",
            "## Artifacts",
            f"- Original: {result.artifacts.original_image}",
            f"- Mask: {result.artifacts.mask_image}",
            f"- Overlay: {result.artifacts.overlay_image}",
            f"- JSON: {result.artifacts.report_json}",
            f"- Log: {result.artifacts.log_file}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def relative_artifact(path: Path) -> str:
    """Return a project-relative artifact path."""

    return to_project_relative_str(path)
