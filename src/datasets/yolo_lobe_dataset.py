"""Utilities for exporting curated nucleus lobe masks to YOLO segmentation format."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from skimage.measure import approximate_polygon, find_contours

from src.core.paths import PROJECT_ROOT, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import NucleusLobeSample, load_lobe_manifest_samples

YOLO_LOBE_CLASS_ID = 0
YOLO_LOBE_CLASS_NAME = "nucleus_lobe"


@dataclass(frozen=True)
class YoloLobeExportRow:
    """One exported YOLO segmentation sample."""

    image_id: str
    split: str
    source_image_path: Path
    yolo_image_path: Path
    yolo_label_path: Path
    segment_count: int
    exported_polygon_count: int

    def to_row(self) -> dict[str, str | int]:
        """Return a CSV row with project-relative paths."""

        return {
            "image_id": self.image_id,
            "split": self.split,
            "source_image_path": to_project_relative_str(self.source_image_path),
            "yolo_image_path": to_project_relative_str(self.yolo_image_path),
            "yolo_label_path": to_project_relative_str(self.yolo_label_path),
            "segment_count": self.segment_count,
            "exported_polygon_count": self.exported_polygon_count,
        }


def polygon_area_xy(points: list[tuple[float, float]]) -> float:
    """Return polygon area for ``(x, y)`` points using the shoelace formula."""

    if len(points) < 3:
        return 0.0
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def _simplified_contour_to_xy_points(
    contour: np.ndarray,
    mask_shape: tuple[int, int],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Convert a padded skimage contour into image-space ``(x, y)`` points."""

    simplified = approximate_polygon(contour, tolerance=tolerance)
    points: list[tuple[float, float]] = []
    height, width = mask_shape
    for row, col in simplified:
        x = float(np.clip(col - 1.0, 0.0, width - 1.0))
        y = float(np.clip(row - 1.0, 0.0, height - 1.0))
        if not points or points[-1] != (x, y):
            points.append((x, y))

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def binary_mask_to_largest_polygon(
    mask: np.ndarray,
    tolerance: float = 2.0,
    min_area_px: float = 12.0,
) -> list[tuple[float, float]] | None:
    """Convert one object mask into its largest polygon contour."""

    if mask.ndim != 2:
        raise ValueError("Object mask must be a 2D array.")
    if not np.any(mask):
        return None

    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant")
    polygons: list[list[tuple[float, float]]] = []
    for contour in find_contours(padded, level=0.5):
        points = _simplified_contour_to_xy_points(
            contour=contour,
            mask_shape=mask.shape,
            tolerance=tolerance,
        )
        if len(points) >= 3 and polygon_area_xy(points) >= min_area_px:
            polygons.append(points)

    if not polygons:
        return None
    return max(polygons, key=polygon_area_xy)


def normalize_polygon_for_yolo(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Normalize image-space polygon points into YOLO segmentation coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    values: list[float] = []
    for x, y in points:
        values.append(float(np.clip(x / image_width, 0.0, 1.0)))
        values.append(float(np.clip(y / image_height, 0.0, 1.0)))
    return values


def instance_mask_to_yolo_lines(
    instance_mask: np.ndarray,
    image_width: int,
    image_height: int,
    tolerance: float = 2.0,
    min_area_px: float = 12.0,
) -> list[str]:
    """Convert a lobe instance mask into YOLO segmentation label lines."""

    if instance_mask.ndim == 3:
        instance_mask = instance_mask[:, :, 0]
    lines: list[str] = []
    for label_value in sorted(int(value) for value in np.unique(instance_mask) if value != 0):
        polygon = binary_mask_to_largest_polygon(
            instance_mask == label_value,
            tolerance=tolerance,
            min_area_px=min_area_px,
        )
        if polygon is None:
            continue
        normalized = normalize_polygon_for_yolo(
            polygon,
            image_width=image_width,
            image_height=image_height,
        )
        if len(normalized) < 6:
            continue
        coordinates = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{YOLO_LOBE_CLASS_ID} {coordinates}")
    return lines


def _copy_image(source: Path, destination: Path, overwrite: bool) -> None:
    """Copy one image into the YOLO dataset tree."""

    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_label(lines: list[str], destination: Path, overwrite: bool) -> None:
    """Write one YOLO label file."""

    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_export_manifest(rows: list[YoloLobeExportRow], path: Path) -> None:
    """Write a manifest for the exported YOLO dataset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "split",
        "source_image_path",
        "yolo_image_path",
        "yolo_label_path",
        "segment_count",
        "exported_polygon_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def write_yolo_data_yaml(output_dir: Path, class_name: str = YOLO_LOBE_CLASS_NAME) -> Path:
    """Write Ultralytics dataset YAML."""

    data = {
        "path": output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {YOLO_LOBE_CLASS_ID: class_name},
    }
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def prepare_yolo_lobe_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    tolerance: float = 2.0,
    min_area_px: float = 12.0,
    overwrite: bool = False,
) -> list[YoloLobeExportRow]:
    """Export curated lobe masks into an Ultralytics YOLO segmentation dataset."""

    output_path = Path(output_dir).resolve()
    samples = load_lobe_manifest_samples(manifest_path=manifest_path)
    rows: list[YoloLobeExportRow] = []
    for sample in samples:
        rows.append(
            _export_sample(
                sample=sample,
                output_dir=output_path,
                tolerance=tolerance,
                min_area_px=min_area_px,
                overwrite=overwrite,
            )
        )

    _write_export_manifest(rows, output_path / "manifest.csv")
    write_yolo_data_yaml(output_path)
    return rows


def _export_sample(
    sample: NucleusLobeSample,
    output_dir: Path,
    tolerance: float,
    min_area_px: float,
    overwrite: bool,
) -> YoloLobeExportRow:
    """Export one curated lobe sample."""

    with Image.open(sample.image_path) as image:
        image_width, image_height = image.size
    with Image.open(sample.lobe_instance_mask_path) as mask_image:
        instance_mask = np.asarray(mask_image)

    label_lines = instance_mask_to_yolo_lines(
        instance_mask=instance_mask,
        image_width=image_width,
        image_height=image_height,
        tolerance=tolerance,
        min_area_px=min_area_px,
    )
    image_path = output_dir / "images" / sample.split / sample.image_path.name
    label_path = output_dir / "labels" / sample.split / f"{sample.image_path.stem}.txt"
    _copy_image(sample.image_path, image_path, overwrite=overwrite)
    _write_label(label_lines, label_path, overwrite=overwrite)

    return YoloLobeExportRow(
        image_id=sample.image_id,
        split=sample.split,
        source_image_path=sample.image_path,
        yolo_image_path=image_path,
        yolo_label_path=label_path,
        segment_count=sample.segment_count,
        exported_polygon_count=len(label_lines),
    )


def project_relative_or_absolute(path: Path) -> str:
    """Return a readable path for CLI output."""

    try:
        return to_project_relative_str(path)
    except ValueError:
        return path.resolve().as_posix().replace(PROJECT_ROOT.as_posix(), ".")
