"""Datasets for binary neutrophil nucleus segmentation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.core.paths import PROJECT_ROOT

SegmentationTransform = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class NucleusSegmentationSample:
    """One image/mask training pair."""

    image_id: str
    image_path: Path
    mask_path: Path
    split: str


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve a path from a manifest row."""

    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (PROJECT_ROOT / path).resolve()


def _load_manifest_samples(
    manifest_path: Path,
    split: str | None,
) -> list[NucleusSegmentationSample]:
    """Load image/mask samples from a curated manifest CSV."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    samples: list[NucleusSegmentationSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"image_id", "image_path", "mask_path", "split"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Manifest is missing required columns: {missing}")

        for row in reader:
            row_split = row["split"]
            if split is not None and row_split != split:
                continue
            samples.append(
                NucleusSegmentationSample(
                    image_id=row["image_id"],
                    image_path=_resolve_path(row["image_path"], manifest_path.parent),
                    mask_path=_resolve_path(row["mask_path"], manifest_path.parent),
                    split=row_split,
                )
            )

    if not samples:
        split_suffix = f" for split '{split}'" if split else ""
        raise ValueError(f"No nucleus segmentation samples found{split_suffix}.")

    missing_files = [
        path
        for sample in samples
        for path in (sample.image_path, sample.mask_path)
        if not path.is_file()
    ]
    if missing_files:
        preview = "\n".join(f"- {path}" for path in missing_files[:5])
        raise FileNotFoundError(f"Manifest references missing files:\n{preview}")
    return samples


def _image_to_tensor(image: np.ndarray) -> Tensor:
    """Convert an RGB image array into a float tensor."""

    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


def _mask_to_tensor(mask: np.ndarray) -> Tensor:
    """Convert a binary mask array into a 1xHxW float tensor."""

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary_mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


class NucleusSegmentationDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """PyTorch dataset for binary nucleus segmentation."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        transform: SegmentationTransform | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_path: Curated dataset manifest with image_path/mask_path columns.
            split: Optional split filter, for example ``train`` or ``val``.
            transform: Optional albumentations transform accepting image and mask.
        """

        self.manifest_path = Path(manifest_path).resolve()
        self.samples = _load_manifest_samples(self.manifest_path, split=split)
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        """Return image tensor, binary mask tensor, and image id."""

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
        with Image.open(sample.mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))

        if self.transform is not None:
            transformed = self.transform(image=rgb_image, mask=mask)
            rgb_image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(rgb_image, np.ndarray):
            image_tensor = _image_to_tensor(rgb_image)
        elif isinstance(rgb_image, torch.Tensor):
            image_tensor = rgb_image.float()
        else:
            raise TypeError("Transform must return image as numpy.ndarray or torch.Tensor.")

        if isinstance(mask, np.ndarray):
            mask_tensor = _mask_to_tensor(mask)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.max() > 1:
                mask_tensor = mask_tensor.div(255.0)
        else:
            raise TypeError("Transform must return mask as numpy.ndarray or torch.Tensor.")

        return image_tensor, (mask_tensor > 0.5).float(), sample.image_id
