"""Оценивает обученные веса U-Net для сегментации ядра."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, PROJECT_ROOT, to_project_relative_str
from src.pipeline.artifacts import save_mask, save_overlay
from src.pipeline.segmentation import UNetNucleusSegmenter

DEFAULT_MANIFEST_PATH = (
    PATHS.processed_data / "nucleus_segmentation" / "curated" / "manifest.csv"
)
DEFAULT_WEIGHTS_PATH = PATHS.models / "unet_nucleus.pt"
DEFAULT_OUTPUT_DIR = PATHS.outputs / "unet_nucleus_eval"


@dataclass(frozen=True)
class EvalSample:
    """Один оценочный образец."""
    image_id: str
    image_path: Path
    mask_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Evaluate U-Net nucleus segmentation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    """Разрешает абсолютные пути или пути относительно проекта."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (PROJECT_ROOT / path).resolve()


def load_samples(manifest_path: Path, split: str) -> list[EvalSample]:
    """Загружает оценочные образцы из курируемого манифеста."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    samples: list[EvalSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["split"] != split:
                continue
            samples.append(
                EvalSample(
                    image_id=row["image_id"],
                    image_path=_resolve_path(row["image_path"], manifest_path.parent),
                    mask_path=_resolve_path(row["mask_path"], manifest_path.parent),
                    split=row["split"],
                )
            )

    if not samples:
        raise ValueError(f"No samples found for split: {split}")
    return samples


def load_rgb_image(path: Path) -> np.ndarray:
    """Загружает массив RGB-изображения."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_binary_mask(path: Path) -> np.ndarray:
    """Загружает массив бинарной маски."""
    return np.asarray(Image.open(path).convert("L")) > 0


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Вычисляет метрику Dice."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Вычисляет пересечение по объединению."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float((intersection + eps) / (union + eps))


def save_comparison_overlay(
    rgb_image: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
    path: Path,
) -> Path:
    """Сохраняет цветное наложение: зеленый - истинное срабатывание, красный - пропуск, синий - ложное срабатывание."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb_image.copy().astype(np.float32)
    target = target_mask.astype(bool)
    prediction = prediction_mask.astype(bool)
    true_positive = target & prediction
    false_negative = target & ~prediction
    false_positive = ~target & prediction

    colors = np.zeros_like(overlay)
    colors[true_positive] = [30, 220, 80]
    colors[false_negative] = [240, 50, 50]
    colors[false_positive] = [50, 120, 255]
    colored_pixels = true_positive | false_negative | false_positive
    overlay[colored_pixels] = 0.55 * overlay[colored_pixels] + 0.45 * colors[colored_pixels]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)
    return path


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Формирует сводку метрик по образцам."""
    dice_values = np.asarray([row["dice"] for row in rows], dtype=np.float32)
    iou_values = np.asarray([row["iou"] for row in rows], dtype=np.float32)
    return {
        "sample_count": len(rows),
        "dice_mean": float(dice_values.mean()),
        "dice_std": float(dice_values.std()),
        "dice_min": float(dice_values.min()),
        "dice_max": float(dice_values.max()),
        "iou_mean": float(iou_values.mean()),
        "iou_std": float(iou_values.std()),
        "iou_min": float(iou_values.min()),
        "iou_max": float(iou_values.max()),
        "worst_samples": sorted(rows, key=lambda row: row["dice"])[:5],
    }


def evaluate(
    samples: list[EvalSample],
    segmenter: UNetNucleusSegmenter,
    output_dir: Path,
) -> dict[str, Any]:
    """Запускает оценку и сохраняет артефакты."""
    predictions_dir = output_dir / "predictions"
    overlays_dir = output_dir / "overlays"
    comparisons_dir = output_dir / "comparisons"
    rows: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="evaluate"):
        rgb_image = load_rgb_image(sample.image_path)
        target_mask = load_binary_mask(sample.mask_path)
        prediction_mask = segmenter.segment(rgb_image)

        prediction_path = save_mask(prediction_mask, predictions_dir / f"{sample.image_id}.png")
        overlay_path = save_overlay(
            rgb_image,
            prediction_mask,
            overlays_dir / f"{sample.image_id}_prediction_overlay.png",
        )
        comparison_path = save_comparison_overlay(
            rgb_image,
            target_mask,
            prediction_mask,
            comparisons_dir / f"{sample.image_id}_comparison.png",
        )

        rows.append(
            {
                "image_id": sample.image_id,
                "split": sample.split,
                "dice": dice_score(prediction_mask, target_mask),
                "iou": iou_score(prediction_mask, target_mask),
                "image_path": to_project_relative_str(sample.image_path),
                "mask_path": to_project_relative_str(sample.mask_path),
                "prediction_path": to_project_relative_str(prediction_path),
                "overlay_path": to_project_relative_str(overlay_path),
                "comparison_path": to_project_relative_str(comparison_path),
            }
        )

    summary = summarize_metrics(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"summary": summary, "samples": rows}


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    samples = load_samples(args.manifest, split=args.split)
    segmenter = UNetNucleusSegmenter(
        weights_path=args.weights_path,
        threshold=args.threshold,
        device_name=args.device,
    )
    result = evaluate(samples=samples, segmenter=segmenter, output_dir=args.output_dir)
    summary = result["summary"]
    print(f"Evaluated split: {args.split}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Dice mean: {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f}")
    print(f"Dice min/max: {summary['dice_min']:.4f} / {summary['dice_max']:.4f}")
    print(f"IoU mean: {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print("Worst samples:")
    for row in summary["worst_samples"]:
        print(f"- {row['image_id']}: dice={row['dice']:.4f}, iou={row['iou']:.4f}")


if __name__ == "__main__":
    main()
