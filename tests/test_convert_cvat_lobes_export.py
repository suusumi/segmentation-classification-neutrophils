"""Tests for CVAT lobe instance export conversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.convert_cvat_lobes_export import (
    build_lobe_instance_mask,
    convert_cvat_lobes_export,
)


def _write_rgb_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 220, 210)).save(path)


def _write_coco_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [
            {"id": 1, "file_name": "cell_a.jpg", "width": 32, "height": 24},
            {"id": 2, "file_name": "cell_b.jpg", "width": 32, "height": 24},
            {"id": 3, "file_name": "cell_c.jpg", "width": 32, "height": 24},
        ],
        "categories": [
            {"id": 1, "name": "nucleus_lobe"},
            {"id": 2, "name": "ignore"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[4, 4, 13, 4, 13, 13, 4, 13]],
            },
            {
                "id": 102,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[17, 4, 26, 4, 26, 13, 17, 13]],
            },
            {
                "id": 201,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [[6, 6, 20, 6, 20, 18, 6, 18]],
            },
            {
                "id": 301,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[4, 5, 10, 5, 10, 12, 4, 12]],
            },
            {
                "id": 302,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[12, 5, 18, 5, 18, 12, 12, 12]],
            },
            {
                "id": 303,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[20, 5, 28, 5, 28, 12, 20, 12]],
            },
            {
                "id": 999,
                "image_id": 3,
                "category_id": 2,
                "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3]],
            },
        ],
    }
    path.write_text(json.dumps(coco), encoding="utf-8")


def test_build_lobe_instance_mask_rasterizes_separate_polygons() -> None:
    annotations = [
        {
            "id": 2,
            "segmentation": [[4, 4, 10, 4, 10, 10, 4, 10]],
        },
        {
            "id": 1,
            "segmentation": [[14, 4, 20, 4, 20, 10, 14, 10]],
        },
    ]

    mask = build_lobe_instance_mask(32, 24, annotations)

    assert mask.dtype == np.uint16
    assert set(np.unique(mask)) == {0, 1, 2}
    assert mask[6, 16] == 1
    assert mask[6, 6] == 2


def test_build_lobe_instance_mask_rejects_coco_rle() -> None:
    annotations = [{"id": 1, "segmentation": {"counts": "encoded", "size": [24, 32]}}]

    with pytest.raises(ValueError, match="RLE"):
        build_lobe_instance_mask(32, 24, annotations)


def test_convert_cvat_lobes_export_writes_curated_manifest(tmp_path: Path) -> None:
    coco_json = tmp_path / "cvat" / "annotations" / "instances_default.json"
    source_dir = tmp_path / "raw" / "neutrophil"
    output_dir = tmp_path / "curated"
    _write_coco_json(coco_json)
    for image_id in ("cell_a", "cell_b", "cell_c"):
        _write_rgb_image(source_dir / f"{image_id}.jpg")

    rows = convert_cvat_lobes_export(
        cvat_dir=tmp_path / "cvat",
        coco_json=coco_json,
        source_images_dir=source_dir,
        output_dir=output_dir,
        val_fraction=1 / 3,
        test_fraction=1 / 3,
        seed=1,
    )

    assert len(rows) == 3
    assert (output_dir / "manifest.csv").is_file()
    assert all(row.image_path.is_file() for row in rows)
    assert all(row.nucleus_mask_path.is_file() for row in rows)
    assert all(row.lobe_instance_mask_path.is_file() for row in rows)
    assert {row.split for row in rows} == {"train", "val", "test"}
    assert {row.image_id: row.segment_count for row in rows} == {
        "cell_a": 2,
        "cell_b": 1,
        "cell_c": 3,
    }

    with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))

    assert manifest_rows[0]["segment_count"] in {"1", "2", "3"}
    cell_a = next(row for row in rows if row.image_id == "cell_a")
    instance_mask = np.asarray(Image.open(cell_a.lobe_instance_mask_path))
    nucleus_mask = np.asarray(Image.open(cell_a.nucleus_mask_path))
    assert set(np.unique(instance_mask)) == {0, 1, 2}
    assert set(np.unique(nucleus_mask)) == {0, 255}
