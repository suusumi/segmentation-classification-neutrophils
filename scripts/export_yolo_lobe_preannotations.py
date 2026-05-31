"""Export YOLO lobe predictions as a CVAT-importable COCO instance ZIP."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
LABEL_NAME = "nucleus_lobe"
DEFAULT_WEIGHTS_PATH = PATHS.models / "yolo_lobes_seg.pt"
DEFAULT_OUTPUT_ZIP = PATHS.processed_data / "nucleus_lobes" / "yolo_preannotations.zip"


@dataclass(frozen=True)
class PreannotationImage:
    """One image to preannotate."""

    image_id: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run YOLO-seg on images and export COCO annotations for CVAT review."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="CSV with image_id and image_path columns, for example a CVAT batch manifest.",
    )
    input_group.add_argument("--images-dir", type=Path, help="Directory with images to annotate.")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--min-polygon-area", type=float, default=12.0)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily."""

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from error
    return YOLO


def _project_path(path_value: str) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_images_from_manifest(manifest_path: Path) -> list[PreannotationImage]:
    """Load images from a CSV manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    rows: list[PreannotationImage] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not {"image_id", "image_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain image_id and image_path columns.")
        for row in reader:
            image_path = _project_path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            rows.append(PreannotationImage(image_id=row["image_id"], image_path=image_path))
    if not rows:
        raise ValueError(f"No images found in manifest: {manifest_path}")
    return rows


def load_images_from_dir(images_dir: Path) -> list[PreannotationImage]:
    """Load supported image files from a directory."""

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory does not exist: {images_dir}")
    rows = [
        PreannotationImage(image_id=path.stem, image_path=path)
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not rows:
        raise ValueError(f"No supported images found in: {images_dir}")
    return rows


def polygon_area(points: list[float]) -> float:
    """Return polygon area using the shoelace formula."""

    if len(points) < 6:
        return 0.0
    xs = np.asarray(points[0::2], dtype=np.float64)
    ys = np.asarray(points[1::2], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def _result_polygons(result: Any, min_polygon_area: float) -> list[list[float]]:
    """Extract COCO polygon lists from one Ultralytics result."""

    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    xy_polygons = getattr(masks, "xy", [])
    polygons: list[list[float]] = []
    for xy_polygon in xy_polygons:
        points: list[float] = []
        for x, y in np.asarray(xy_polygon, dtype=np.float64):
            points.extend([round(float(x), 2), round(float(y), 2)])
        if polygon_area(points) >= min_polygon_area:
            polygons.append(points)
    return polygons


def _bbox_from_polygon(points: list[float]) -> list[float]:
    """Return COCO bbox from a flat polygon."""

    xs = np.asarray(points[0::2], dtype=np.float64)
    ys = np.asarray(points[1::2], dtype=np.float64)
    x_min = float(xs.min())
    y_min = float(ys.min())
    return [x_min, y_min, float(xs.max() - x_min), float(ys.max() - y_min)]


def build_coco(rows: list[PreannotationImage], results: list[Any], min_polygon_area: float) -> dict[str, Any]:
    """Build a COCO instance dictionary from YOLO results."""

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_index, (row, result) in enumerate(zip(rows, results, strict=True), start=1):
        with Image.open(row.image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_index,
                "file_name": row.image_path.name,
                "width": width,
                "height": height,
            }
        )
        for polygon in _result_polygons(result, min_polygon_area=min_polygon_area):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": [polygon],
                    "area": polygon_area(polygon),
                    "bbox": _bbox_from_polygon(polygon),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "YOLO-generated nucleus_lobe preannotations for CVAT review.",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": LABEL_NAME, "supercategory": "nucleus"}],
    }


def write_coco_zip(coco: dict[str, Any], output_zip: Path) -> Path:
    """Write a COCO annotations zip importable by CVAT."""

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "annotations/instances_default.json",
            json.dumps(coco, indent=2),
        )
    return output_zip


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    if not args.weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights file does not exist: {args.weights_path}")

    rows = (
        load_images_from_manifest(args.manifest)
        if args.manifest is not None
        else load_images_from_dir(args.images_dir)
    )
    yolo_class = _load_yolo_class()
    model = yolo_class(args.weights_path.as_posix())
    predict_kwargs: dict[str, Any] = {
        "source": [row.image_path.as_posix() for row in rows],
        "imgsz": args.image_size,
        "conf": args.conf,
        "iou": args.iou,
        "stream": False,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device
    results = list(model.predict(**predict_kwargs))
    coco = build_coco(rows=rows, results=results, min_polygon_area=args.min_polygon_area)
    output_zip = write_coco_zip(coco, args.output_zip)
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Output ZIP: {to_project_relative_str(output_zip)}")


if __name__ == "__main__":
    main()
