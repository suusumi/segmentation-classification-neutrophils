"""Datasets for nucleus lobe count baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.core.paths import PROJECT_ROOT

LobeTransform = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class NucleusLobeSample:
    """One curated nucleus lobe sample."""

    image_id: str
    source_image_path: Path
    image_path: Path
    nucleus_mask_path: Path
    lobe_instance_mask_path: Path
    segment_count: int
    split: str


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve absolute, manifest-relative, or project-relative paths."""

    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (PROJECT_ROOT / path).resolve()


def load_lobe_manifest_samples(
    manifest_path: str | Path,
    split: str | None = None,
) -> list[NucleusLobeSample]:
    """Load curated lobe samples from a manifest CSV."""

    resolved_manifest_path = Path(manifest_path).resolve()
    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {resolved_manifest_path}")

    samples: list[NucleusLobeSample] = []
    with resolved_manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "image_id",
            "source_image_path",
            "image_path",
            "nucleus_mask_path",
            "lobe_instance_mask_path",
            "segment_count",
            "split",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Manifest is missing required columns: {missing}")

        for row in reader:
            row_split = row["split"]
            if split is not None and row_split != split:
                continue
            samples.append(
                NucleusLobeSample(
                    image_id=row["image_id"],
                    source_image_path=_resolve_path(
                        row["source_image_path"],
                        resolved_manifest_path.parent,
                    ),
                    image_path=_resolve_path(row["image_path"], resolved_manifest_path.parent),
                    nucleus_mask_path=_resolve_path(
                        row["nucleus_mask_path"],
                        resolved_manifest_path.parent,
                    ),
                    lobe_instance_mask_path=_resolve_path(
                        row["lobe_instance_mask_path"],
                        resolved_manifest_path.parent,
                    ),
                    segment_count=int(row["segment_count"]),
                    split=row_split,
                )
            )

    if not samples:
        split_suffix = f" for split '{split}'" if split else ""
        raise ValueError(f"No nucleus lobe samples found{split_suffix}.")

    missing_files = [
        path
        for sample in samples
        for path in (
            sample.image_path,
            sample.nucleus_mask_path,
            sample.lobe_instance_mask_path,
        )
        if not path.is_file()
    ]
    if missing_files:
        preview = "\n".join(f"- {path}" for path in missing_files[:5])
        raise FileNotFoundError(f"Manifest references missing files:\n{preview}")
    return samples


def load_lobe_count_values(manifest_path: str | Path) -> list[int]:
    """Return sorted segment count values present in the full manifest."""

    samples = load_lobe_manifest_samples(manifest_path=manifest_path, split=None)
    return sorted({sample.segment_count for sample in samples})


def _image_to_tensor(image: np.ndarray) -> Tensor:
    """Convert an RGB image array into a float tensor."""

    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


def _mask_to_tensor(mask: np.ndarray | Tensor) -> Tensor:
    """Convert a mask array or tensor into a 1xHxW float tensor."""

    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

    mask_tensor = mask.float()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.max() > 1:
        mask_tensor = mask_tensor.div(float(mask_tensor.max()))
    return (mask_tensor > 0).float()


class NucleusLobeCountDataset(Dataset[tuple[Tensor, Tensor, Tensor, str]]):
    """Dataset for count baselines using RGB image plus nucleus mask."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        count_values: Sequence[int] | None = None,
        transform: LobeTransform | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_path: Curated lobe manifest CSV.
            split: Optional split filter.
            count_values: Stable ordered count labels, for example ``[2, 3, 4, 5]``.
            transform: Optional albumentations transform accepting image and masks.
        """

        self.manifest_path = Path(manifest_path).resolve()
        self.samples = load_lobe_manifest_samples(self.manifest_path, split=split)
        self.count_values = list(count_values or sorted({s.segment_count for s in self.samples}))
        self.count_to_index = {count: index for index, count in enumerate(self.count_values)}
        self.transform = transform

        unknown_counts = sorted(
            {sample.segment_count for sample in self.samples}.difference(self.count_to_index)
        )
        if unknown_counts:
            raise ValueError(f"Dataset contains counts not present in count_values: {unknown_counts}")

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        """Return 4-channel input tensor, class target, raw count, and image id."""

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
        with Image.open(sample.nucleus_mask_path) as mask_image:
            nucleus_mask = np.asarray(mask_image.convert("L"))
        with Image.open(sample.lobe_instance_mask_path) as instance_image:
            lobe_instance_mask = np.asarray(instance_image)

        if self.transform is not None:
            transformed = self.transform(
                image=rgb_image,
                mask=nucleus_mask,
                lobe_instance_mask=lobe_instance_mask,
            )
            rgb_image = transformed["image"]
            nucleus_mask = transformed["mask"]

        if isinstance(rgb_image, np.ndarray):
            image_tensor = _image_to_tensor(rgb_image)
        elif isinstance(rgb_image, torch.Tensor):
            image_tensor = rgb_image.float()
        else:
            raise TypeError("Transform must return image as numpy.ndarray or torch.Tensor.")

        mask_tensor = _mask_to_tensor(nucleus_mask)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0)
        target = torch.tensor(self.count_to_index[sample.segment_count], dtype=torch.long)
        segment_count = torch.tensor(sample.segment_count, dtype=torch.long)
        return input_tensor, target, segment_count, sample.image_id
