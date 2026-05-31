"""Tests for nucleus lobe count baseline utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.nucleus_lobe_count_dataset import (
    NucleusLobeCountDataset,
    load_lobe_count_values,
)
from src.datasets.transforms import get_lobe_count_val_transforms
from src.models import LobeCountCNN
from src.training.lobe_count_metrics import compute_lobe_count_metrics


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(230, 220, 210)).save(path)


def _write_binary_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[6:18, 8:24] = 255
    Image.fromarray(mask).save(path)


def _write_instance_mask(path: Path, segment_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint16)
    for index in range(segment_count):
        col = 3 + index * 6
        mask[6:14, col : col + 4] = index + 1
    Image.fromarray(mask).save(path)


def _write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
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
        writer.writerows(rows)


def _create_dataset_fixture(tmp_path: Path) -> Path:
    rows: list[dict[str, str | int]] = []
    for image_id, segment_count, split in (
        ("cell_a", 2, "train"),
        ("cell_b", 3, "val"),
        ("cell_c", 5, "test"),
    ):
        image_path = tmp_path / "images" / split / f"{image_id}.jpg"
        nucleus_mask_path = tmp_path / "nucleus_masks" / split / f"{image_id}.png"
        instance_mask_path = tmp_path / "lobe_instance_masks" / split / f"{image_id}.png"
        _write_rgb_image(image_path)
        _write_binary_mask(nucleus_mask_path)
        _write_instance_mask(instance_mask_path, segment_count=segment_count)
        rows.append(
            {
                "image_id": image_id,
                "source_image_path": str(image_path),
                "image_path": str(image_path),
                "nucleus_mask_path": str(nucleus_mask_path),
                "lobe_instance_mask_path": str(instance_mask_path),
                "segment_count": segment_count,
                "split": split,
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_lobe_count_dataset_returns_four_channel_input(tmp_path: Path) -> None:
    manifest_path = _create_dataset_fixture(tmp_path)
    count_values = load_lobe_count_values(manifest_path)
    dataset = NucleusLobeCountDataset(
        manifest_path=manifest_path,
        split="train",
        count_values=count_values,
        transform=get_lobe_count_val_transforms(image_size=16),
    )

    inputs, target, segment_count, image_id = dataset[0]

    assert inputs.shape == (4, 16, 16)
    assert target.item() == 0
    assert segment_count.item() == 2
    assert image_id == "cell_a"
    assert set(count_values) == {2, 3, 5}


def test_lobe_count_model_forward_shape() -> None:
    model = LobeCountCNN(in_channels=4, num_classes=4, base_channels=4)
    logits = model(torch.rand(2, 4, 64, 64))

    assert logits.shape == (2, 4)


def test_lobe_count_metrics_include_count_and_hypersegmentation_accuracy() -> None:
    metrics = compute_lobe_count_metrics(
        targets=[2, 3, 5, 5],
        predictions=[2, 4, 4, 5],
        count_values=[2, 3, 4, 5],
        hypersegmentation_threshold=5,
    )

    assert metrics.sample_count == 4
    assert metrics.exact_accuracy == 0.5
    assert metrics.plus_minus_one_accuracy == 1.0
    assert metrics.binary_hypersegmentation_accuracy == 0.75
    assert metrics.confusion_matrix == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
    ]
