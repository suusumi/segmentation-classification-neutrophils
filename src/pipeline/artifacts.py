"""Artifact writers for masks, overlays, and reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.core.paths import to_project_relative_str
from src.pipeline.result import AnalysisResult
from src.utils.io import write_json


def _format_report_value(value: object) -> str:
    """Format report values for Markdown tables."""

    if value is None:
        return "не сформирован"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(rows: list[tuple[str, object]]) -> list[str]:
    """Build a two-column Markdown table."""

    table = [
        "| Параметр | Значение |",
        "| --- | --- |",
    ]
    table.extend(f"| {name} | {_format_report_value(value)} |" for name, value in rows)
    return table


def _classification_summary(result: AnalysisResult) -> str:
    """Return a concise Russian explanation for the classification result."""

    segments = result.features.nucleus_segments
    segment_word = _russian_segment_word(segments)
    if result.classification.label == "unknown":
        return "Ядро не было надежно обнаружено, поэтому класс не определен."
    if result.classification.label == "hypersegmentation":
        return f"Обнаружено {segments} {segment_word} ядра; это соответствует гиперсегментации."
    return f"Обнаружено {segments} {segment_word} ядра; это ниже порога гиперсегментации."


def _russian_segment_word(count: int) -> str:
    """Return the correct Russian plural form for a segment count."""

    if count % 10 == 1 and count % 100 != 11:
        return "сегмент"
    if count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14}:
        return "сегмента"
    return "сегментов"


def _localized_status(status: str) -> str:
    """Return a Russian label for a pipeline status."""

    return {
        "completed": "завершен",
        "failed": "ошибка",
        "running": "выполняется",
    }.get(status, status)


def _localized_classification(label: str) -> str:
    """Return a Russian label for a classification result."""

    return {
        "normal": "норма",
        "hypersegmentation": "гиперсегментация",
        "unknown": "не определено",
    }.get(label, label)


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


def save_labeled_mask(mask: np.ndarray, path: Path) -> Path:
    """Save labeled components with a deterministic color palette."""

    path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(
        [
            [0, 0, 0],
            [220, 30, 30],
            [30, 160, 220],
            [40, 190, 90],
            [230, 180, 40],
            [160, 90, 230],
            [240, 100, 160],
            [70, 210, 190],
        ],
        dtype=np.uint8,
    )
    colored = palette[np.mod(mask.astype(np.int32), len(palette))]
    Image.fromarray(colored).save(path)
    return path


def save_lobe_overlay(
    rgb_image: np.ndarray,
    split_mask: np.ndarray,
    boundary_mask: np.ndarray,
    path: Path,
) -> Path:
    """Save colored lobe components and white predicted boundaries over the image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray(rgb_image.astype(np.uint8)).convert("RGBA")
    palette = np.asarray(
        [
            [0, 0, 0, 0],
            [220, 30, 30, 120],
            [30, 160, 220, 120],
            [40, 190, 90, 120],
            [230, 180, 40, 120],
            [160, 90, 230, 120],
            [240, 100, 160, 120],
            [70, 210, 190, 120],
        ],
        dtype=np.uint8,
    )
    overlay = palette[np.mod(split_mask.astype(np.int32), len(palette))]
    overlay[boundary_mask.astype(bool)] = [255, 255, 255, 210]
    Image.alpha_composite(base, Image.fromarray(overlay)).save(path)
    return path


def save_report_json(result: AnalysisResult, path: Path) -> Path:
    """Persist the complete result as JSON."""

    write_json(path, result.to_dict())
    return path


def save_report_markdown(result: AnalysisResult, path: Path) -> Path:
    """Persist a compact human-readable report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    postprocessing_labels = {
        "segment_min_area_px": "Мин. площадь сегмента, px",
        "segment_min_peak_distance_px": "Мин. расстояние между пиками, px",
        "watershed_compactness": "Компактность watershed",
        "lobe_foreground_threshold": "Порог маски долей",
        "lobe_boundary_threshold": "Порог границ долей",
        "lobe_min_segment_area_px": "Мин. площадь доли, px",
        "yolo_lobe_confidence": "Порог уверенности YOLO",
        "yolo_lobe_iou": "YOLO IoU",
        "yolo_lobe_image_size": "YOLO размер изображения",
        "yolo_lobe_min_mask_area_px": "YOLO мин. площадь маски, px",
        "border_margin_px": "Отступ от края, px",
        "cluster_distance_fraction": "Дистанция кластеров",
        "cluster_min_area_ratio": "Мин. доля площади кластера",
    }
    feature_rows = [
        ("Площадь ядра, px", result.features.nucleus_area_px),
        ("Периметр ядра, px", result.features.nucleus_perimeter_px),
        ("Округлость", result.features.nucleus_circularity),
        ("Плотность / solidity", result.features.nucleus_solidity),
        ("Эксцентриситет", result.features.nucleus_eccentricity),
        ("Заполнение ограничивающего прямоугольника", result.features.nucleus_extent),
        ("Ориентация, градусы", result.features.nucleus_orientation_degrees),
        ("Большая ось, px", result.features.nucleus_major_axis_length_px),
        ("Малая ось, px", result.features.nucleus_minor_axis_length_px),
        ("Соотношение сторон", result.features.nucleus_aspect_ratio),
        ("Выпуклая площадь, px", result.features.nucleus_convex_area_px),
        ("Заполненная площадь, px", result.features.nucleus_filled_area_px),
        ("Ширина ограничивающего прямоугольника, px", result.features.nucleus_bbox_width_px),
        ("Высота ограничивающего прямоугольника, px", result.features.nucleus_bbox_height_px),
        ("Сегменты ядра", result.features.nucleus_segments),
        ("Средняя площадь сегмента, px", result.features.segment_area_mean_px),
        ("Минимальная площадь сегмента, px", result.features.segment_area_min_px),
        ("Максимальная площадь сегмента, px", result.features.segment_area_max_px),
        ("Доля foreground", result.features.mask_foreground_fraction),
    ]
    pipeline_rows = [
        ("Версия пайплайна", result.metadata.pipeline_version),
        ("Сегментатор ядра", result.metadata.segmenter_name),
        ("Счетчик долей", result.metadata.lobe_counter_name),
        ("Классификатор", result.metadata.classifier_name),
    ]
    postprocessing_rows = [
        (postprocessing_labels.get(str(key), str(key)), value)
        for key, value in result.metadata.postprocessing.items()
    ]
    artifact_rows = [
        ("Исходное изображение", result.artifacts.original_image),
        ("Маска ядра", result.artifacts.mask_image),
        ("Наложение маски", result.artifacts.overlay_image),
        ("Маска долей", result.artifacts.lobe_foreground_image),
        ("Границы долей", result.artifacts.lobe_boundary_image),
        ("Компоненты долей", result.artifacts.lobe_components_image),
        ("Наложение долей", result.artifacts.lobe_overlay_image),
        ("JSON-отчет", result.artifacts.report_json),
        ("Markdown-отчет", result.artifacts.report_markdown),
        ("Лог анализа", result.artifacts.log_file),
    ]

    content = "\n".join(
        [
            f"# Отчет анализа {result.analysis_id}",
            "",
            "## Сводка",
            "",
            *_markdown_table(
                [
                    ("Статус", _localized_status(result.status)),
                    ("Входной файл", result.input_filename),
                    ("Размер изображения", f"{result.image_width_px}x{result.image_height_px}px"),
                    ("Класс", _localized_classification(result.classification.label)),
                    ("Техническая оценка классификатора", result.classification.confidence),
                    ("Обоснование", _classification_summary(result)),
                ]
            ),
            "",
            "## Детали анализа",
            "",
            "### Морфология ядра",
            "",
            *_markdown_table(feature_rows),
            "",
            "### Конфигурация пайплайна",
            "",
            *_markdown_table(pipeline_rows),
            "",
            "### Параметры постобработки",
            "",
            *_markdown_table(postprocessing_rows),
            "",
            "### Артефакты",
            "",
            *_markdown_table(artifact_rows),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def relative_artifact(path: Path) -> str:
    """Return a project-relative artifact path."""

    return to_project_relative_str(path)
