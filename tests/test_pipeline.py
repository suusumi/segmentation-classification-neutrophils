"""Tests for the single-image neutrophil analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.pipeline.pipeline import NeutrophilAnalysisPipeline


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

    result = NeutrophilAnalysisPipeline().run(
        analysis_id="test-analysis",
        image_path=image_path,
        output_dir=output_dir,
    )

    assert result.status == "completed"
    assert result.features.nucleus_area_px > 0
    assert result.classification.label in {"normal", "hypersegmentation", "unknown"}
    assert (output_dir / "artifacts" / "nucleus_mask.png").is_file()
    assert (output_dir / "artifacts" / "overlay.png").is_file()
    assert (output_dir / "reports" / "result.json").is_file()
    assert (output_dir / "reports" / "report.md").is_file()

