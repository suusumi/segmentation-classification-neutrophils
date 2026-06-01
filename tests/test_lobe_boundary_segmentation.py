"""Tests for lobe boundary segmentation baseline utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.nucleus_lobe_segmentation_dataset import (
    NucleusLobeSegmentationDataset,
    lobe_instance_mask_to_targets,
)
from src.datasets.transforms import get_lobe_segmentation_val_transforms
from src.models.unet import UNet
from src.pipeline.lobe_counting import UNetLobeBoundaryCounter
from src.pipeline.lobe_segmentation_postprocessing import postprocess_lobe_segmentation


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(230, 220, 210)).save(path)


def _write_binary_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[6:18, 4:28] = 255
    Image.fromarray(mask).save(path)


def _write_instance_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint16)
    mask[6:18, 4:14] = 1
    mask[6:18, 14:24] = 2
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


def _write_predicted_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[7:17, 5:27] = 255
    Image.fromarray(mask).save(path)


def test_instance_mask_to_targets_builds_foreground_and_boundary() -> None:
    instance_mask = np.zeros((12, 16), dtype=np.uint16)
    instance_mask[3:9, 3:8] = 1
    instance_mask[3:9, 8:13] = 2

    targets = lobe_instance_mask_to_targets(instance_mask, boundary_radius=0)

    assert targets.shape == (2, 12, 16)
    assert targets[0].sum() == 60
    assert targets[1].sum() > 0
    assert targets[1, 4, 7] == 1


def test_instance_mask_to_targets_builds_separator_for_nearby_gapped_instances() -> None:
    instance_mask = np.zeros((24, 32), dtype=np.uint16)
    instance_mask[7:17, 5:13] = 1
    instance_mask[7:17, 18:26] = 2

    targets = lobe_instance_mask_to_targets(
        instance_mask,
        boundary_radius=2,
        boundary_gap_radius=4,
    )

    assert targets[1].sum() > 0
    assert targets[1, 12, 12] == 1
    assert targets[1, 12, 18] == 1


def test_lobe_segmentation_dataset_returns_input_and_two_channel_target(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "train" / "cell.jpg"
    nucleus_path = tmp_path / "nucleus_masks" / "train" / "cell.png"
    instance_path = tmp_path / "lobe_instance_masks" / "train" / "cell.png"
    manifest_path = tmp_path / "manifest.csv"
    _write_rgb_image(image_path)
    _write_binary_mask(nucleus_path)
    _write_instance_mask(instance_path)
    _write_manifest(manifest_path, image_path, nucleus_path, instance_path)

    dataset = NucleusLobeSegmentationDataset(
        manifest_path=manifest_path,
        split="train",
        transform=get_lobe_segmentation_val_transforms(image_size=16),
    )

    inputs, targets, segment_count, image_id = dataset[0]

    assert inputs.shape == (4, 16, 16)
    assert targets.shape == (2, 16, 16)
    assert segment_count.item() == 2
    assert image_id == "cell"


def test_lobe_segmentation_dataset_can_use_predicted_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "train" / "cell.jpg"
    nucleus_path = tmp_path / "nucleus_masks" / "train" / "cell.png"
    instance_path = tmp_path / "lobe_instance_masks" / "train" / "cell.png"
    predicted_dir = tmp_path / "predicted"
    manifest_path = tmp_path / "manifest.csv"
    _write_rgb_image(image_path)
    _write_binary_mask(nucleus_path)
    _write_instance_mask(instance_path)
    _write_predicted_mask(predicted_dir / "train" / "cell.png")
    _write_manifest(manifest_path, image_path, nucleus_path, instance_path)

    dataset = NucleusLobeSegmentationDataset(
        manifest_path=manifest_path,
        split="train",
        nucleus_mask_source="predicted",
        predicted_mask_dir=predicted_dir,
    )

    inputs, _, _, _ = dataset[0]

    assert int(inputs[3].sum().item()) == 220


def test_postprocess_lobe_segmentation_counts_split_components() -> None:
    foreground = np.zeros((32, 32), dtype=np.float32)
    foreground[8:24, 6:26] = 1.0
    boundary = np.zeros((32, 32), dtype=np.float32)
    boundary[8:24, 15:17] = 1.0

    result = postprocess_lobe_segmentation(
        foreground_probability=foreground,
        boundary_probability=boundary,
        min_segment_area_px=8,
    )

    assert result.segment_count == 2
    assert set(np.unique(result.split_mask)) == {0, 1, 2}
    assert np.count_nonzero(result.split_mask) == np.count_nonzero(result.foreground_mask)


def test_postprocess_lobe_segmentation_respects_support_mask() -> None:
    foreground = np.ones((12, 16), dtype=np.float32)
    boundary = np.zeros((12, 16), dtype=np.float32)
    support = np.zeros((12, 16), dtype=bool)
    support[3:9, 4:12] = True

    result = postprocess_lobe_segmentation(
        foreground,
        boundary,
        min_segment_area_px=4,
        support_mask=support,
    )

    assert result.segment_count == 1
    assert result.foreground_mask.sum() == support.sum()
    assert np.count_nonzero(result.split_mask) == support.sum()


def test_unet_lobe_boundary_counter_loads_checkpoint(tmp_path: Path) -> None:
    weights_path = tmp_path / "lobe_boundary_unet.pt"
    model = UNet(in_channels=4, out_channels=2, base_channels=4)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": 4,
                "out_channels": 2,
                "base_channels": 4,
                "image_size": 16,
            },
            "postprocessing": {
                "foreground_threshold": 0.5,
                "boundary_threshold": 0.5,
                "min_segment_area_px": 4,
            },
        },
        weights_path,
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    nucleus_mask = np.zeros((24, 32), dtype=bool)
    nucleus_mask[6:18, 8:24] = True

    result = UNetLobeBoundaryCounter(
        weights_path=weights_path,
        device_name="cpu",
    ).count(image, nucleus_mask)

    assert result.foreground_mask is not None
    assert result.boundary_mask is not None
    assert result.split_mask is not None
    assert result.foreground_mask.shape == nucleus_mask.shape
    assert result.segment_count.segment_count >= 0
