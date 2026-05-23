"""End-to-end neutrophil image analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from src.core.paths import PATHS
from src.core.settings import SETTINGS
from src.pipeline.artifacts import (
    relative_artifact,
    save_mask,
    save_overlay,
    save_report_json,
    save_report_markdown,
)
from src.pipeline.classification import classify_neutrophil
from src.pipeline.features import extract_morphological_features
from src.pipeline.postprocessing import postprocess_mask
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.result import AnalysisArtifacts, AnalysisResult, PipelineMetadata
from src.pipeline.segment_counting import SegmentCountConfig, count_nucleus_segments
from src.pipeline.segmentation import ThresholdNucleusSegmenter, UNetNucleusSegmenter
from src.services.errors import PipelineError
from src.utils.logger import get_logger


class NeutrophilAnalysisPipeline:
    """Pipeline for single-neutrophil images."""

    version = "0.2.0"

    def __init__(self, segmenter_name: str = SETTINGS.segmenter_name) -> None:
        self.segmenter_name = segmenter_name
        self.segment_count_config = SegmentCountConfig()

    def _build_segmenter(self):
        if self.segmenter_name == "threshold":
            return ThresholdNucleusSegmenter()
        if self.segmenter_name == "unet":
            return UNetNucleusSegmenter(weights_path=SETTINGS.unet_weights_path)
        raise PipelineError(f"Unknown segmenter: {self.segmenter_name}")

    def run(
        self,
        analysis_id: str,
        image_path: Path,
        output_dir: Path | None = None,
        input_filename: str | None = None,
    ) -> AnalysisResult:
        """Run the complete analysis pipeline and write artifacts."""

        analysis_dir = output_dir or PATHS.analyses / analysis_id
        artifacts_dir = analysis_dir / "artifacts"
        reports_dir = analysis_dir / "reports"
        logs_dir = analysis_dir / "logs"
        for directory in (artifacts_dir, reports_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / "analysis.log"
        logger = get_logger(f"analysis.{analysis_id}", log_file=log_path)
        logger.info("Starting analysis %s for %s", analysis_id, image_path)

        preprocessed = preprocess_image(image_path)
        segmenter = self._build_segmenter()
        raw_mask = segmenter.segment(preprocessed.normalized_rgb)
        mask = postprocess_mask(raw_mask)
        segment_count = count_nucleus_segments(mask, config=self.segment_count_config)
        features = extract_morphological_features(mask, segment_count=segment_count)
        classification = classify_neutrophil(features)

        mask_path = save_mask(mask, artifacts_dir / "nucleus_mask.png")
        overlay_path = save_overlay(preprocessed.rgb, mask, artifacts_dir / "overlay.png")
        report_json_path = reports_dir / "result.json"
        report_markdown_path = reports_dir / "report.md"

        result = AnalysisResult(
            analysis_id=analysis_id,
            status="completed",
            input_filename=input_filename or image_path.name,
            image_width_px=preprocessed.width,
            image_height_px=preprocessed.height,
            features=features,
            classification=classification,
            artifacts=AnalysisArtifacts(
                original_image=relative_artifact(image_path),
                mask_image=relative_artifact(mask_path),
                overlay_image=relative_artifact(overlay_path),
                report_json=relative_artifact(report_json_path),
                report_markdown=relative_artifact(report_markdown_path),
                log_file=relative_artifact(log_path),
            ),
            metadata=PipelineMetadata(
                pipeline_version=self.version,
                segmenter_name=segmenter.name,
                classifier_name="rule_based_segment_count",
                postprocessing={
                    "segment_min_area_px": self.segment_count_config.min_segment_area_px,
                    "segment_min_peak_distance_px": (
                        self.segment_count_config.min_peak_distance_px
                    ),
                    "watershed_compactness": self.segment_count_config.watershed_compactness,
                },
            ),
        )
        save_report_json(result, report_json_path)
        save_report_markdown(result, report_markdown_path)

        logger.info(
            "Analysis %s completed with label=%s segments=%s",
            analysis_id,
            result.classification.label,
            result.features.nucleus_segments,
        )
        return result
