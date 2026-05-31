"""Tests for the single-image neutrophil analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.pipeline.classification import classify_neutrophil
from src.pipeline.features import extract_morphological_features
from src.pipeline.pipeline import NeutrophilAnalysisPipeline
from src.pipeline.postprocessing import postprocess_mask
from src.pipeline.segment_counting import SegmentCountConfig, count_nucleus_segments


def _create_synthetic_neutrophil(path: Path) -> None:
    """Create a simple RGB image with a dark segmented nucleus."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (128, 128), color=(235, 218, 215))
    draw = ImageDraw.Draw(image)
    draw.ellipse((36, 48, 58, 72), fill=(52, 44, 116))
    draw.ellipse((70, 48, 92, 72), fill=(52, 44, 116))
    image.save(path)


def test_pipeline_writes_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "cell.png"
    output_dir = tmp_path / "analysis"
    _create_synthetic_neutrophil(image_path)

    result = NeutrophilAnalysisPipeline(
        segmenter_name="threshold",
        lobe_counter_name="watershed",
    ).run(
        analysis_id="test-analysis",
        image_path=image_path,
        output_dir=output_dir,
    )

    assert result.status == "completed"
    assert result.features.nucleus_area_px > 0
    assert result.features.nucleus_circularity > 0
    assert result.features.segment_area_mean_px > 0
    assert result.classification.label in {"normal", "hypersegmentation", "unknown"}
    assert result.metadata.segmenter_name == "threshold"
    assert result.metadata.lobe_counter_name == "watershed"
    assert result.metadata.classifier_name == "rule_based_segment_count"
    assert (output_dir / "artifacts" / "nucleus_mask.png").is_file()
    assert (output_dir / "artifacts" / "overlay.png").is_file()
    assert (output_dir / "artifacts" / "lobe_components.png").is_file()
    assert (output_dir / "artifacts" / "lobe_overlay.png").is_file()
    assert (output_dir / "reports" / "result.json").is_file()
    assert (output_dir / "reports" / "report.md").is_file()


def test_pipeline_can_use_provided_nucleus_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "cell.png"
    mask_path = tmp_path / "input" / "mask.png"
    output_dir = tmp_path / "analysis"
    _create_synthetic_neutrophil(image_path)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[48:82, 36:92] = 255
    Image.fromarray(mask).save(mask_path)

    result = NeutrophilAnalysisPipeline(
        segmenter_name="threshold",
        lobe_counter_name="watershed",
    ).run_with_nucleus_mask(
        analysis_id="test-lobe-debug",
        image_path=image_path,
        nucleus_mask_path=mask_path,
        output_dir=output_dir,
    )

    assert result.metadata.segmenter_name == "provided_nucleus_mask"
    assert result.metadata.lobe_counter_name == "watershed"
    assert result.features.nucleus_area_px > 0
    assert (output_dir / "artifacts" / "nucleus_mask.png").is_file()


def test_segment_counting_splits_components() -> None:
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:36, 20:36] = True
    mask[20:36, 58:74] = True
    mask[58:74, 38:54] = True

    result = count_nucleus_segments(
        mask,
        config=SegmentCountConfig(min_segment_area_px=32, min_peak_distance_px=6),
    )

    assert result.segment_count == 3
    assert result.segment_area_min_px > 0
    assert result.segment_area_max_px >= result.segment_area_min_px


def test_postprocessing_removes_border_artifacts() -> None:
    mask = np.zeros((96, 96), dtype=bool)
    mask[2:10, 2:10] = True
    mask[40:58, 40:58] = True

    cleaned = postprocess_mask(mask, border_margin_px=8)

    assert not cleaned[2:10, 2:10].any()
    assert cleaned[40:58, 40:58].any()


def test_postprocessing_removes_distant_same_color_artifacts() -> None:
    mask = np.zeros((128, 128), dtype=bool)
    mask[52:76, 52:76] = True
    mask[58:78, 78:98] = True
    mask[104:112, 104:112] = True
    mask[108:116, 12:20] = True

    cleaned = postprocess_mask(mask)

    assert cleaned[52:76, 52:76].any()
    assert cleaned[58:78, 78:98].any()
    assert not cleaned[104:112, 104:112].any()
    assert not cleaned[108:116, 12:20].any()


def test_features_and_classifier_detect_hypersegmentation() -> None:
    mask = np.zeros((128, 128), dtype=bool)
    for row, col in ((20, 20), (20, 74), (56, 44), (86, 18), (86, 78)):
        mask[row : row + 18, col : col + 18] = True

    segment_count = count_nucleus_segments(
        mask,
        config=SegmentCountConfig(min_segment_area_px=32, min_peak_distance_px=6),
    )
    features = extract_morphological_features(mask, segment_count=segment_count)
    classification = classify_neutrophil(features)

    assert features.nucleus_segments == 5
    assert features.mask_foreground_fraction > 0
    assert features.nucleus_aspect_ratio >= 1.0
    assert classification.label == "hypersegmentation"
