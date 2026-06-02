"""Тесты подготовки пакета CVAT."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_cvat_batch import (
    load_manifest_rows,
    prepare_cvat_batch,
    sample_rows,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(230, 220, 210)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 5:11] = 255
    Image.fromarray(mask).save(path)


def _write_manifest(path: Path, image_path: Path, mask_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "source_image_path",
                "image_path",
                "pseudo_mask_path",
                "overlay_path",
                "split",
                "annotation_status",
                "needs_review",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": image_path.stem,
                "source_image_path": image_path.as_posix(),
                "image_path": image_path.as_posix(),
                "pseudo_mask_path": mask_path.as_posix(),
                "overlay_path": "",
                "split": "train",
                "annotation_status": "pseudo",
                "needs_review": "true",
            }
        )


def test_prepare_cvat_batch_writes_images_and_segmentation_mask_zip(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "SNE_test.jpg"
    mask_path = tmp_path / "pseudo_masks" / "SNE_test.png"
    manifest_path = tmp_path / "manifest.csv"
    _write_image(image_path)
    _write_mask(mask_path)
    _write_manifest(manifest_path, image_path, mask_path)

    rows = sample_rows(load_manifest_rows(manifest_path), limit=1, seed=42)
    batch_dir = prepare_cvat_batch(rows, output_dir=tmp_path / "cvat_batches", batch_name="batch")

    assert (batch_dir / "manifest.csv").is_file()
    assert (batch_dir / "labels.json").is_file()
    assert (batch_dir / "images.zip").is_file()
    assert (batch_dir / "preannotations_segmentation_mask.zip").is_file()

    with zipfile.ZipFile(batch_dir / "images.zip") as archive:
        assert archive.namelist() == ["SNE_test.jpg"]

    with zipfile.ZipFile(batch_dir / "preannotations_segmentation_mask.zip") as archive:
        names = set(archive.namelist())
        assert "labelmap.txt" in names
        assert "ImageSets/Segmentation/default.txt" in names
        assert "SegmentationClass/SNE_test.png" in names
        with archive.open("SegmentationClass/SNE_test.png") as file:
            mask = np.asarray(Image.open(file))

    assert mask.max() == 1
    assert mask.min() == 0
