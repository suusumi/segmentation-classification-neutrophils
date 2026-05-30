"""Generate CVAT COCO preannotations for nucleus lobe review."""

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
from skimage.measure import approximate_polygon
from skimage.measure import find_contours
from skimage.measure import regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.pipeline.pipeline import NeutrophilAnalysisPipeline
from src.pipeline.postprocessing import postprocess_mask
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.segment_counting import count_nucleus_segments

DEFAULT_BATCH_DIR = PATHS.processed_data / "nucleus_segmentation" / "cvat_batches" / "batch_001"
DEFAULT_OUTPUT_ZIP = (
    PATHS.processed_data
    / "nucleus_lobes"
    / "cvat_batches"
    / "batch_001"
    / "preannotations_coco_instances.zip"
)
LABEL_NAME = "nucleus_lobe"


@dataclass(frozen=True)
class BatchImage:
    """One image from a CVAT batch manifest."""

    image_id: str
    image_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate COCO instance preannotations for nucleus_lobe review."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help="CVAT batch directory containing manifest.csv and images.zip.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help="Output COCO zip importable into CVAT.",
    )
    parser.add_argument(
        "--segmenter",
        default="auto",
        choices=("auto", "unet", "threshold"),
        help="Nucleus segmenter used before watershed splitting.",
    )
    parser.add_argument(
        "--min-polygon-area",
        type=float,
        default=48.0,
        help="Skip tiny contour polygons below this area in pixels.",
    )
    parser.add_argument(
        "--polygon-tolerance",
        type=float,
        default=2.0,
        help="Polygon simplification tolerance in pixels.",
    )
    return parser.parse_args()


def _project_path(path_value: str) -> Path:
    """Resolve a project-relative path value."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_batch_images(manifest_path: Path) -> list[BatchImage]:
    """Load image paths from a CVAT batch manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Batch manifest does not exist: {manifest_path}")

    rows: list[BatchImage] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = _project_path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            rows.append(
                BatchImage(
                    image_id=row["image_id"],
                    image_path=image_path,
                    split=row.get("split", "unsplit"),
                )
            )

    if not rows:
        raise ValueError(f"No rows found in batch manifest: {manifest_path}")
    return rows


def polygon_area(points: list[float]) -> float:
    """Return polygon area using the shoelace formula."""

    if len(points) < 6:
        return 0.0
    xs = np.asarray(points[0::2], dtype=np.float64)
    ys = np.asarray(points[1::2], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def mask_to_polygons(
    mask: np.ndarray,
    tolerance: float,
    min_area: float,
) -> list[list[float]]:
    """Convert one binary object mask into COCO polygon lists."""

    polygons: list[list[float]] = []
    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant")
    contours = find_contours(padded, level=0.5)
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = approximate_polygon(contour, tolerance=tolerance)
        if len(simplified) < 3:
            continue

        points: list[float] = []
        for row, col in simplified:
            x = float(np.clip(col - 1.0, 0.0, mask.shape[1] - 1.0))
            y = float(np.clip(row - 1.0, 0.0, mask.shape[0] - 1.0))
            points.extend([round(x, 2), round(y, 2)])

        if polygon_area(points) >= min_area:
            polygons.append(points)
    return polygons


def build_preannotation_coco(
    rows: list[BatchImage],
    segmenter_name: str,
    polygon_tolerance: float,
    min_polygon_area: float,
) -> dict[str, Any]:
    """Run the current pipeline split logic and return COCO annotations."""

    pipeline = NeutrophilAnalysisPipeline(segmenter_name=segmenter_name)
    segmenter = pipeline._build_segmenter()
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_index, row in enumerate(rows, start=1):
        preprocessed = preprocess_image(row.image_path)
        raw_mask = segmenter.segment(preprocessed.normalized_rgb)
        nucleus_mask = postprocess_mask(raw_mask)
        segments = count_nucleus_segments(nucleus_mask, config=pipeline.segment_count_config)
        images.append(
            {
                "id": image_index,
                "file_name": row.image_path.name,
                "width": preprocessed.width,
                "height": preprocessed.height,
            }
        )

        for region in regionprops(segments.labeled_mask):
            object_mask = segments.labeled_mask == region.label
            polygons = mask_to_polygons(
                object_mask,
                tolerance=polygon_tolerance,
                min_area=min_polygon_area,
            )
            if not polygons:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": float(region.area),
                    "bbox": [
                        float(region.bbox[1]),
                        float(region.bbox[0]),
                        float(region.bbox[3] - region.bbox[1]),
                        float(region.bbox[2] - region.bbox[0]),
                    ],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "Auto-generated nucleus_lobe preannotations for CVAT review.",
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
    rows = load_batch_images(args.batch_dir / "manifest.csv")
    coco = build_preannotation_coco(
        rows=rows,
        segmenter_name=args.segmenter,
        polygon_tolerance=args.polygon_tolerance,
        min_polygon_area=args.min_polygon_area,
    )
    output_zip = write_coco_zip(coco, args.output_zip)
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Output ZIP: {to_project_relative_str(output_zip)}")


if __name__ == "__main__":
    main()
