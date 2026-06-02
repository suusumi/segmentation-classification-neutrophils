"""Оценивает модель сегментации YOLO для подсчета долей ядра."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import load_lobe_count_values, load_lobe_manifest_samples
from src.training.lobe_count_metrics import compute_lobe_count_metrics

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_WEIGHTS_PATH = PATHS.models / "yolo_lobes_seg.pt"
DEFAULT_OUTPUT_DIR = PATHS.outputs / "yolo_lobes_eval"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Evaluate YOLO-seg lobe instance predictions.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--min-mask-area-px", type=int, default=16)
    parser.add_argument("--hypersegmentation-threshold", type=int, default=5)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Лениво импортирует Ultralytics."""

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from error
    return YOLO


def _mask_areas(result: Any) -> list[int]:
    """Возвращает площади предсказанных масок из одного результата Ultralytics."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    data = getattr(masks, "data", None)
    if data is None:
        return []
    array = data.detach().cpu().numpy()
    if array.ndim != 3:
        return []
    return [int(mask.astype(bool).sum()) for mask in array]


def _mean_confidence(result: Any) -> float:
    """Возвращает среднюю уверенность рамок для одного результата Ultralytics."""
    boxes = getattr(result, "boxes", None)
    confidences = getattr(boxes, "conf", None) if boxes is not None else None
    if confidences is None or len(confidences) == 0:
        return 0.0
    return float(confidences.detach().cpu().numpy().mean())


def _save_plot(result: Any, path: Path) -> str:
    """Сохраняет один отрисованный график предсказания Ultralytics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    Image.fromarray(np.asarray(plotted)).save(path)
    return to_project_relative_str(path)


def _predict(model: Any, image_paths: list[Path], args: argparse.Namespace) -> list[Any]:
    """Запускает предсказание YOLO по путям изображений."""
    predict_kwargs: dict[str, Any] = {
        "source": [path.as_posix() for path in image_paths],
        "imgsz": args.image_size,
        "conf": args.conf,
        "iou": args.iou,
        "stream": False,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device
    return list(model.predict(**predict_kwargs))


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Записывает JSON с метриками и CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    if not args.weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights file does not exist: {args.weights_path}")

    samples = load_lobe_manifest_samples(args.manifest, split=args.split)
    yolo_class = _load_yolo_class()
    model = yolo_class(args.weights_path.as_posix())
    results = _predict(model=model, image_paths=[sample.image_path for sample in samples], args=args)
    if len(results) != len(samples):
        raise ValueError("YOLO returned a different number of results than requested images.")

    rows: list[dict[str, Any]] = []
    targets: list[int] = []
    predictions: list[int] = []
    for sample, result in zip(samples, results, strict=True):
        mask_areas = _mask_areas(result)
        kept_areas = [area for area in mask_areas if area >= args.min_mask_area_px]
        predicted_count = len(kept_areas)
        target_count = sample.segment_count
        row: dict[str, Any] = {
            "image_id": sample.image_id,
            "target_count": target_count,
            "predicted_count": predicted_count,
            "absolute_error": abs(predicted_count - target_count),
            "raw_mask_count": len(mask_areas),
            "mean_confidence": _mean_confidence(result),
            "image_path": to_project_relative_str(sample.image_path),
        }
        if args.save_plots:
            row["plot_path"] = _save_plot(result, args.output_dir / "plots" / f"{sample.image_id}.jpg")
        rows.append(row)
        targets.append(target_count)
        predictions.append(predicted_count)

    count_values = sorted(set(load_lobe_count_values(args.manifest)).union(predictions))
    metrics = compute_lobe_count_metrics(
        targets=targets,
        predictions=predictions,
        count_values=count_values,
        hypersegmentation_threshold=args.hypersegmentation_threshold,
    )
    summary = asdict(metrics)
    summary["split"] = args.split
    summary["weights_path"] = to_project_relative_str(args.weights_path)
    summary["image_size"] = args.image_size
    summary["conf"] = args.conf
    summary["iou"] = args.iou
    summary["min_mask_area_px"] = args.min_mask_area_px
    write_outputs(args.output_dir, summary=summary, rows=rows)

    print(f"Evaluated split: {args.split}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Exact count accuracy: {summary['exact_accuracy']:.4f}")
    print(f"+/-1 accuracy: {summary['plus_minus_one_accuracy']:.4f}")
    print(
        "Binary hypersegmentation accuracy: "
        f"{summary['binary_hypersegmentation_accuracy']:.4f}"
    )
    print(f"Confusion matrix labels: {summary['count_values']}")
    print(f"Confusion matrix: {summary['confusion_matrix']}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")


if __name__ == "__main__":
    main()
