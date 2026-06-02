"""Сквозной пайплайн анализа изображений нейтрофилов."""
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


def _mask_summary(mask: np.ndarray) -> tuple[int, float]:
    """Возвращает количество и долю пикселей переднего плана для компактного логирования."""
    foreground_px = int(mask.astype(bool).sum())
    total_px = int(mask.size)
    foreground_fraction = float(foreground_px / total_px) if total_px else 0.0
    return foreground_px, foreground_fraction


def _load_nucleus_mask(mask_path: Path, height: int, width: int) -> np.ndarray:
    """Загружает бинарную маску ядра и при необходимости меняет размер методом ближайшего соседа."""
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


class NeutrophilAnalysisPipeline:
    """Пайплайн для изображений одиночных нейтрофилов."""
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
        """Запускает полный пайплайн анализа и записывает артефакты."""
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
        """Запускает подсчет долей с переданной маской ядра."""
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
        """Запускает полный пайплайн анализа и записывает артефакты."""
        analysis_dir = output_dir or PATHS.analyses / analysis_id
        artifacts_dir = analysis_dir / "artifacts"
        reports_dir = analysis_dir / "reports"
        logs_dir = analysis_dir / "logs"
        for directory in (artifacts_dir, reports_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / "analysis.log"
        logger = get_logger(f"analysis.{analysis_id}", log_file=log_path)
        logger.info(
            "Starting analysis %s input=%s output_dir=%s",
            analysis_id,
            image_path,
            analysis_dir,
        )
        logger.info(
            "Runtime configuration segmenter=%s lobe_counter=%s",
            self.segmenter_name,
            self.lobe_counter_name,
        )

        preprocessed = preprocess_image(image_path)
        logger.info(
            "Preprocessed image width=%s height=%s normalized_dtype=%s",
            preprocessed.width,
            preprocessed.height,
            preprocessed.normalized_rgb.dtype,
        )
        if nucleus_mask_path is None:
            segmenter = self._build_segmenter()
            logger.info("Selected nucleus segmenter name=%s", segmenter.name)
            raw_mask = segmenter.segment(preprocessed.normalized_rgb)
            segmenter_name = segmenter.name
        else:
            logger.info("Using provided nucleus mask path=%s", nucleus_mask_path)
            raw_mask = _load_nucleus_mask(nucleus_mask_path, preprocessed.height, preprocessed.width)
            segmenter_name = segmenter_label or "provided_nucleus_mask"
        raw_foreground_px, raw_foreground_fraction = _mask_summary(raw_mask)
        logger.info(
            "Raw nucleus mask foreground_px=%s foreground_fraction=%.4f",
            raw_foreground_px,
            raw_foreground_fraction,
        )
        mask = postprocess_mask(raw_mask)
        mask_foreground_px, mask_foreground_fraction = _mask_summary(mask)
        logger.info(
            "Postprocessed nucleus mask foreground_px=%s foreground_fraction=%.4f removed_px=%s",
            mask_foreground_px,
            mask_foreground_fraction,
            raw_foreground_px - mask_foreground_px,
        )
        lobe_counter = self._build_lobe_counter()
        logger.info("Selected lobe counter name=%s", lobe_counter.name)
        lobe_result = lobe_counter.count(preprocessed.normalized_rgb, mask)
        segment_count = lobe_result.segment_count
        logger.info(
            "Lobe count result segments=%s area_min_px=%s area_mean_px=%.2f area_max_px=%s",
            segment_count.segment_count,
            segment_count.segment_area_min_px,
            segment_count.segment_area_mean_px,
            segment_count.segment_area_max_px,
        )
        if lobe_result.foreground_mask is not None:
            foreground_px, foreground_fraction = _mask_summary(lobe_result.foreground_mask)
            logger.info(
                "Lobe foreground mask foreground_px=%s foreground_fraction=%.4f",
                foreground_px,
                foreground_fraction,
            )
        if lobe_result.boundary_mask is not None:
            boundary_px, boundary_fraction = _mask_summary(lobe_result.boundary_mask)
            logger.info(
                "Lobe boundary mask foreground_px=%s foreground_fraction=%.4f",
                boundary_px,
                boundary_fraction,
            )
        features = extract_morphological_features(mask, segment_count=segment_count)
        logger.info(
            "Extracted features nucleus_area_px=%s circularity=%.3f solidity=%.3f "
            "aspect_ratio=%.3f mask_fraction=%.4f",
            features.nucleus_area_px,
            features.nucleus_circularity,
            features.nucleus_solidity,
            features.nucleus_aspect_ratio,
            features.mask_foreground_fraction,
        )
        classification = classify_neutrophil(features)
        logger.info(
            "Classification label=%s confidence=%.3f reason=%s",
            classification.label,
            classification.confidence,
            classification.reason,
        )

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
            "Artifacts saved mask=%s overlay=%s lobe_components=%s lobe_overlay=%s report_json=%s",
            mask_path,
            overlay_path,
            lobe_components_path,
            lobe_overlay_path,
            report_json_path,
        )

        logger.info(
            "Analysis %s completed with label=%s segments=%s",
            analysis_id,
            result.classification.label,
            result.features.nucleus_segments,
        )
        return result
