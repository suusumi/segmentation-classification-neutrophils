"""Тесты экспорта набора данных долей ядра для YOLO."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.yolo_lobe_dataset import (
    binary_mask_to_largest_polygon,
    instance_mask_to_yolo_lines,
    prepare_yolo_lobe_dataset,
)


def _write_rgb_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 220, 210)).save(path)


def _write_instance_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint16)
    mask[4:12, 4:13] = 1
    mask[13:21, 17:28] = 2
    Image.fromarray(mask).save(path)


def _write_nucleus_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:21, 4:28] = 255
    Image.fromarray(mask).save(path)


def _write_manifest(path: Path, image_path: Path, nucleus_path: Path, instance_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "source_image_path",
                "image_path",
                "nucleus_mask_path",
                "lobe_instance_mask_path",
                "segment_count",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "cell",
                "source_image_path": str(image_path),
                "image_path": str(image_path),
                "nucleus_mask_path": str(nucleus_path),
                "lobe_instance_mask_path": str(instance_path),
                "segment_count": 2,
                "split": "train",
            }
        )


def test_binary_mask_to_largest_polygon_returns_image_space_points() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:12, 4:13] = 1

    polygon = binary_mask_to_largest_polygon(mask, tolerance=1.0, min_area_px=4)

    assert polygon is not None
    assert len(polygon) >= 3
    assert all(0 <= x <= 31 and 0 <= y <= 23 for x, y in polygon)


def test_instance_mask_to_yolo_lines_exports_one_line_per_instance() -> None:
    mask = np.zeros((24, 32), dtype=np.uint16)
    mask[4:12, 4:13] = 1
    mask[13:21, 17:28] = 2

    lines = instance_mask_to_yolo_lines(mask, image_width=32, image_height=24)

    assert len(lines) == 2
    for line in lines:
        values = line.split()
        assert values[0] == "0"
        coordinates = [float(value) for value in values[1:]]
        assert len(coordinates) >= 6
        assert all(0.0 <= value <= 1.0 for value in coordinates)


def test_prepare_yolo_lobe_dataset_writes_labels_and_data_yaml(tmp_path: Path) -> None:
    image_path = tmp_path / "curated" / "images" / "train" / "cell.jpg"
    nucleus_path = tmp_path / "curated" / "nucleus_masks" / "train" / "cell.png"
    instance_path = tmp_path / "curated" / "lobe_instance_masks" / "train" / "cell.png"
    manifest_path = tmp_path / "curated" / "manifest.csv"
    output_dir = tmp_path / "yolo"
    _write_rgb_image(image_path)
    _write_nucleus_mask(nucleus_path)
    _write_instance_mask(instance_path)
    _write_manifest(manifest_path, image_path, nucleus_path, instance_path)

    rows = prepare_yolo_lobe_dataset(
        manifest_path=manifest_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert len(rows) == 1
    assert rows[0].exported_polygon_count == 2
    assert (output_dir / "images" / "train" / "cell.jpg").is_file()
    assert (output_dir / "labels" / "train" / "cell.txt").is_file()
    assert (output_dir / "data.yaml").is_file()
    assert (output_dir / "manifest.csv").is_file()
    assert len((output_dir / "labels" / "train" / "cell.txt").read_text().splitlines()) == 2
