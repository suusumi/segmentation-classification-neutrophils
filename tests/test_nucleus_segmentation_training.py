"""Tests for nucleus segmentation dataset preparation and inference glue."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.convert_cvat_nucleus_export import convert_cvat_export, load_cvat_binary_mask
from src.datasets.nucleus_segmentation_dataset import NucleusSegmentationDataset
from src.datasets.transforms import get_segmentation_val_transforms
from src.models import UNet
from src.pipeline.segmentation import UNetNucleusSegmenter


def _write_rgb_image(path: Path, color: tuple[int, int, int] = (230, 220, 210)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def _write_cvat_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32, 3), dtype=np.uint8)
    mask[6:16, 10:22] = (220, 30, 30)
    Image.fromarray(mask).save(path)


def test_load_cvat_binary_mask_converts_rgb_label_to_binary(tmp_path: Path) -> None:
    mask_path = tmp_path / "SegmentationClass" / "cell.png"
    _write_cvat_mask(mask_path)

    mask = load_cvat_binary_mask(mask_path)

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert mask[8, 12] == 255


def test_convert_cvat_export_writes_curated_manifest(tmp_path: Path) -> None:
    cvat_dir = tmp_path / "cvat"
    source_dir = tmp_path / "raw" / "neutrophil"
    output_dir = tmp_path / "curated"

    ids = ["cell_a", "cell_b", "cell_c"]
    default_path = cvat_dir / "ImageSets" / "Segmentation" / "default.txt"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    for image_id in ids:
        _write_rgb_image(source_dir / f"{image_id}.jpg")
        _write_cvat_mask(cvat_dir / "SegmentationClass" / f"{image_id}.png")

    rows = convert_cvat_export(
        cvat_dir=cvat_dir,
        source_images_dir=source_dir,
        output_dir=output_dir,
        val_fraction=1 / 3,
        test_fraction=1 / 3,
        seed=1,
    )

    assert len(rows) == 3
    assert (output_dir / "manifest.csv").is_file()
    assert all(row.image_path.is_file() for row in rows)
    assert all(row.mask_path.is_file() for row in rows)
    assert {row.split for row in rows} == {"train", "val", "test"}


def test_nucleus_segmentation_dataset_returns_image_and_mask_tensors(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "train" / "cell.jpg"
    mask_path = tmp_path / "masks" / "train" / "cell.png"
    manifest_path = tmp_path / "manifest.csv"
    _write_rgb_image(image_path)
    _write_cvat_mask(mask_path)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "image_path", "mask_path", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "cell",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": "train",
            }
        )

    dataset = NucleusSegmentationDataset(
        manifest_path=manifest_path,
        split="train",
        transform=get_segmentation_val_transforms(image_size=16),
    )

    image, mask, image_id = dataset[0]

    assert image.shape == (3, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert image_id == "cell"
    assert torch.all((mask == 0) | (mask == 1))


def test_unet_segmenter_loads_checkpoint_and_returns_mask(tmp_path: Path) -> None:
    weights_path = tmp_path / "unet_nucleus.pt"
    model = UNet(base_channels=4)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": 3,
                "out_channels": 1,
                "base_channels": 4,
                "image_size": 32,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        weights_path,
    )
    image = np.full((24, 32, 3), fill_value=128, dtype=np.uint8)

    segmenter = UNetNucleusSegmenter(weights_path=weights_path, device_name="cpu")
    mask = segmenter.segment(image)

    assert mask.shape == (24, 32)
    assert mask.dtype == bool
