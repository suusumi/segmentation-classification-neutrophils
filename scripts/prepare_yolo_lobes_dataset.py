"""Подготавливает набор данных сегментации Ultralytics YOLO для долей ядра."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.yolo_lobe_dataset import prepare_yolo_lobe_dataset

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PATHS.processed_data / "nucleus_lobes" / "yolo_seg"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Convert curated nucleus lobe instance masks to YOLO segmentation labels."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--polygon-tolerance",
        type=float,
        default=2.0,
        help="Contour simplification tolerance in pixels.",
    )
    parser.add_argument(
        "--min-polygon-area",
        type=float,
        default=12.0,
        help="Skip object polygons smaller than this area in pixels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    rows = prepare_yolo_lobe_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        tolerance=args.polygon_tolerance,
        min_area_px=args.min_polygon_area,
        overwrite=args.overwrite,
    )
    split_counts = Counter(row.split for row in rows)
    skipped = sum(row.segment_count - row.exported_polygon_count for row in rows)

    print(f"Exported samples: {len(rows)}")
    print(
        "Splits: "
        f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
    )
    print(f"Skipped tiny/invalid polygons: {skipped}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print(f"YOLO data YAML: {to_project_relative_str(args.output_dir / 'data.yaml')}")


if __name__ == "__main__":
    main()
