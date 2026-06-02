"""Обучает модель сегментации YOLO для экземпляров долей ядра."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str

DEFAULT_DATA_YAML = PATHS.processed_data / "nucleus_lobes" / "yolo_seg" / "data.yaml"
DEFAULT_PROJECT_DIR = PATHS.outputs / "yolo_lobes_train"
DEFAULT_WEIGHTS_PATH = PATHS.models / "yolo_lobes_seg.pt"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Train YOLO-seg on nucleus lobe instances.")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Ultralytics segmentation model checkpoint, for example yolo11n-seg.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--name", default="yolo_lobes_seg")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Лениво импортирует Ultralytics, чтобы остальная часть проекта могла работать без него."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from error
    return YOLO


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    if not args.data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO data YAML does not exist: {args.data_yaml}. "
            "Run scripts/prepare_yolo_lobes_dataset.py first."
        )

    yolo_class = _load_yolo_class()
    model = yolo_class(args.model)
    train_kwargs: dict[str, Any] = {
        "data": args.data_yaml.as_posix(),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "project": args.project.as_posix(),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "task": "segment",
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    result = model.train(**train_kwargs)
    save_dir = Path(getattr(result, "save_dir", args.project / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished but best weights were not found: {best_weights}")

    args.weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.weights_path)
    print(f"Run directory: {to_project_relative_str(save_dir)}")
    print(f"Best weights: {to_project_relative_str(best_weights)}")
    print(f"Copied weights: {to_project_relative_str(args.weights_path)}")


if __name__ == "__main__":
    main()
