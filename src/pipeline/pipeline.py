"""End-to-end neutrophil image analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.core.paths import PATHS
from src.core.settings import SETTINGS
from src.pipeline.artifacts import (
    relative_artifact,
    save_labeled_mask,
    save_lobe_overlay,
    save_mask,
    save_overlay,
    save_report_json,
    save_report_markdown,
)
from src.pipeline.classification import classify_neutrophil
from src.pipeline.features import extract_morphological_features
from src.pipeline.lobe_counting import (
    UNetLobeBoundaryCounter,
    WatershedLobeCounter,
    YOLOLobeInstanceCounter,
)
from src.pipeline.postprocessing import (
    DEFAULT_BORDER_MARGIN_PX,
    DEFAULT_CLUSTER_DISTANCE_FRACTION,
    DEFAULT_CLUSTER_MIN_AREA_RATIO,
    postprocess_mask,
)
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.result import AnalysisArtifacts, AnalysisResult, PipelineMetadata
from src.pipeline.segment_counting import SegmentCountConfig
from src.pipeline.segmentation import ThresholdNucleusSegmenter, UNetNucleusSegmenter
from src.services.errors import PipelineError
from src.utils.logger import get_logger


def _load_nucleus_mask(mask_path: Path, height: int, width: int) -> np.ndarray:
    """Load a binary nucleus mask and resize with nearest-neighbor if needed."""

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


class NeutrophilAnalysisPipeline:
    """Pipeline for single-neutrophil images."""

    version = "0.2.0"

    def __init__(
        self,
        segmenter_name: str = SETTINGS.segmenter_name,
        lobe_counter_name: str = SETTINGS.lobe_counter_name,
    ) -> None:
        self.segmenter_name = segmenter_name
        self.lobe_counter_name = lobe_counter_name
        self.segment_count_config = SegmentCountConfig()

    def _build_segmenter(self):
        segmenter_name = self.segmenter_name.strip().lower()
        if segmenter_name == "auto":
            if SETTINGS.unet_weights_path.is_file():
                return UNetNucleusSegmenter(weights_path=SETTINGS.unet_weights_path)
            return ThresholdNucleusSegmenter()
        if segmenter_name == "threshold":
            return ThresholdNucleusSegmenter()
        if segmenter_name == "unet":
            return UNetNucleusSegmenter(weights_path=SETTINGS.unet_weights_path)
        raise PipelineError(f"Unknown segmenter: {self.segmenter_name}")

    def _build_lobe_counter(self):
        lobe_counter_name = self.lobe_counter_name.strip().lower()
        if lobe_counter_name == "auto":
            if SETTINGS.lobe_boundary_weights_path.is_file():
                return UNetLobeBoundaryCounter(
                    weights_path=SETTINGS.lobe_boundary_weights_path,
                    foreground_threshold=SETTINGS.lobe_foreground_threshold,
                    boundary_threshold=SETTINGS.lobe_boundary_threshold,
                    min_segment_area_px=SETTINGS.lobe_min_segment_area_px,
                )
            return WatershedLobeCounter(config=self.segment_count_config)
        if lobe_counter_name == "watershed":
            return WatershedLobeCounter(config=self.segment_count_config)
        if lobe_counter_name in {"lobe_boundary_unet", "unet"}:
            return UNetLobeBoundaryCounter(
                weights_path=SETTINGS.lobe_boundary_weights_path,
                foreground_threshold=SETTINGS.lobe_foreground_threshold,
                boundary_threshold=SETTINGS.lobe_boundary_threshold,
                min_segment_area_px=SETTINGS.lobe_min_segment_area_px,
            )
        if lobe_counter_name in {"yolo", "yolo_lobes", "yolo_lobes_seg"}:
            return YOLOLobeInstanceCounter(
                weights_path=SETTINGS.yolo_lobe_weights_path,
                confidence=SETTINGS.yolo_lobe_confidence,
                iou=SETTINGS.yolo_lobe_iou,
                image_size=SETTINGS.yolo_lobe_image_size,
                min_mask_area_px=SETTINGS.yolo_lobe_min_mask_area_px,
                device_name=SETTINGS.yolo_lobe_device,
            )
        raise PipelineError(f"Unknown lobe counter: {self.lobe_counter_name}")

    def run(
        self,
        analysis_id: str,
        image_path: Path,
        output_dir: Path | None = None,
        input_filename: str | None = None,
    ) -> AnalysisResult:
        """Run the complete analysis pipeline and write artifacts."""

        return self._run(
            analysis_id=analysis_id,
            image_path=image_path,
            output_dir=output_dir,
            input_filename=input_filename,
            nucleus_mask_path=None,
            segmenter_label=None,
        )

    def run_with_nucleus_mask(
        self,
        analysis_id: str,
        image_path: Path,
        nucleus_mask_path: Path,
        output_dir: Path | None = None,
        input_filename: str | None = None,
    ) -> AnalysisResult:
        """Run lobe counting with a provided nucleus mask."""

        return self._run(
            analysis_id=analysis_id,
            image_path=image_path,
            output_dir=output_dir,
            input_filename=input_filename,
            nucleus_mask_path=nucleus_mask_path,
            segmenter_label="provided_nucleus_mask",
        )

    def _run(
        self,
        analysis_id: str,
        image_path: Path,
        output_dir: Path | None,
        input_filename: str | None,
        nucleus_mask_path: Path | None,
        segmenter_label: str | None,
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
        if nucleus_mask_path is None:
            segmenter = self._build_segmenter()
            raw_mask = segmenter.segment(preprocessed.normalized_rgb)
            segmenter_name = segmenter.name
        else:
            raw_mask = _load_nucleus_mask(nucleus_mask_path, preprocessed.height, preprocessed.width)
            segmenter_name = segmenter_label or "provided_nucleus_mask"
        mask = postprocess_mask(raw_mask)
        lobe_counter = self._build_lobe_counter()
        lobe_result = lobe_counter.count(preprocessed.normalized_rgb, mask)
        segment_count = lobe_result.segment_count
        features = extract_morphological_features(mask, segment_count=segment_count)
        classification = classify_neutrophil(features)

        mask_path = save_mask(mask, artifacts_dir / "nucleus_mask.png")
        overlay_path = save_overlay(preprocessed.rgb, mask, artifacts_dir / "overlay.png")
        lobe_foreground_path = None
        lobe_boundary_path = None
        lobe_components_path = None
        lobe_overlay_path = None
        if lobe_result.foreground_mask is not None:
            lobe_foreground_path = save_mask(
                lobe_result.foreground_mask,
                artifacts_dir / "lobe_foreground.png",
            )
        if lobe_result.boundary_mask is not None:
            lobe_boundary_path = save_mask(
                lobe_result.boundary_mask,
                artifacts_dir / "lobe_boundary.png",
            )
        if lobe_result.split_mask is not None:
            lobe_components_path = save_labeled_mask(
                lobe_result.split_mask,
                artifacts_dir / "lobe_components.png",
            )
            boundary_mask = (
                lobe_result.boundary_mask
                if lobe_result.boundary_mask is not None
                else np.zeros(lobe_result.split_mask.shape, dtype=bool)
            )
            lobe_overlay_path = save_lobe_overlay(
                preprocessed.rgb,
                lobe_result.split_mask,
                boundary_mask,
                artifacts_dir / "lobe_overlay.png",
            )
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
                lobe_foreground_image=(
                    relative_artifact(lobe_foreground_path) if lobe_foreground_path else None
                ),
                lobe_boundary_image=(
                    relative_artifact(lobe_boundary_path) if lobe_boundary_path else None
                ),
                lobe_components_image=(
                    relative_artifact(lobe_components_path) if lobe_components_path else None
                ),
                lobe_overlay_image=relative_artifact(lobe_overlay_path) if lobe_overlay_path else None,
            ),
            metadata=PipelineMetadata(
                pipeline_version=self.version,
                segmenter_name=segmenter_name,
                lobe_counter_name=lobe_counter.name,
                classifier_name="rule_based_segment_count",
                postprocessing={
                    "segment_min_area_px": self.segment_count_config.min_segment_area_px,
                    "segment_min_peak_distance_px": (
                        self.segment_count_config.min_peak_distance_px
                    ),
                    "watershed_compactness": self.segment_count_config.watershed_compactness,
                    "lobe_foreground_threshold": getattr(
                        lobe_counter,
                        "foreground_threshold",
                        None,
                    ),
                    "lobe_boundary_threshold": getattr(lobe_counter, "boundary_threshold", None),
                    "lobe_min_segment_area_px": getattr(lobe_counter, "min_segment_area_px", None),
                    "yolo_lobe_confidence": getattr(lobe_counter, "confidence", None),
                    "yolo_lobe_iou": getattr(lobe_counter, "iou", None),
                    "yolo_lobe_image_size": getattr(lobe_counter, "image_size", None),
                    "yolo_lobe_min_mask_area_px": getattr(
                        lobe_counter,
                        "min_mask_area_px",
                        None,
                    ),
                    "border_margin_px": DEFAULT_BORDER_MARGIN_PX,
                    "cluster_distance_fraction": DEFAULT_CLUSTER_DISTANCE_FRACTION,
                    "cluster_min_area_ratio": DEFAULT_CLUSTER_MIN_AREA_RATIO,
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
