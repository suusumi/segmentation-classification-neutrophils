"""Datasets for nucleus lobe foreground and boundary segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import torch
from PIL import Image
from skimage.morphology import (
    closing,
    dilation,
    disk,
    erosion,
)
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.nucleus_lobe_count_dataset import (
    NucleusLobeSample,
    load_lobe_manifest_samples,
)

LobeSegmentationTransform = Callable[..., dict[str, Any]]
NucleusMaskSource = Literal["cvat", "predicted", "mixed"]


def _sample_predicted_mask_path(
    predicted_mask_dir: Path,
    sample: NucleusLobeSample,
) -> Path:
    """Return expected predicted mask path for a lobe sample."""

    return predicted_mask_dir / sample.split / f"{sample.image_id}.png"


def _image_to_tensor(image: np.ndarray) -> Tensor:
    """Convert an RGB image array into a float tensor."""

    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div(255.0)


def _mask_to_tensor(mask: np.ndarray | Tensor) -> Tensor:
    """Convert a binary mask array or tensor into a 1xHxW float tensor."""

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


def _instance_mask_to_numpy(mask: np.ndarray | Tensor) -> np.ndarray:
    """Convert an instance mask to a 2D numpy array."""

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint16, copy=False)


def lobe_instance_mask_to_targets(
    instance_mask: np.ndarray | Tensor,
    boundary_radius: int = 1,
    boundary_gap_radius: int = 6,
) -> Tensor:
    """Convert an instance mask into foreground and boundary targets."""

    instance_array = _instance_mask_to_numpy(instance_mask)
    foreground = instance_array > 0
    coverage = np.zeros(instance_array.shape, dtype=np.uint8)
    for label_value in np.unique(instance_array):
        if label_value == 0:
            continue
        coverage += dilation(
            instance_array == label_value,
            footprint=disk(boundary_gap_radius),
        ).astype(np.uint8)
    boundary = coverage >= 2
    if boundary_radius > 0:
        boundary = dilation(boundary, footprint=disk(boundary_radius)) & foreground

    stacked = np.stack(
        [
            foreground.astype(np.float32),
            boundary.astype(np.float32),
        ],
        axis=0,
    )
    return torch.from_numpy(stacked)


class NucleusLobeSegmentationDataset(Dataset[tuple[Tensor, Tensor, Tensor, str]]):
    """Dataset for lobe foreground/boundary segmentation."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        transform: LobeSegmentationTransform | None = None,
        boundary_radius: int = 1,
        nucleus_mask_source: NucleusMaskSource = "cvat",
        predicted_mask_dir: str | Path | None = None,
        predicted_mask_probability: float = 0.5,
        mask_noise_probability: float = 0.0,
        mask_noise_max_radius: int = 2,
        boundary_gap_radius: int = 6,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_path: Curated lobe manifest CSV.
            split: Optional split filter.
            transform: Optional albumentations transform accepting image and masks.
            boundary_radius: Pixel radius used to widen boundary targets.
            nucleus_mask_source: Which nucleus mask input to use: curated CVAT mask,
                generated predicted mask, or a random mix.
            predicted_mask_dir: Root directory with predicted masks stored as
                ``<split>/<image_id>.png``.
            predicted_mask_probability: Probability of using a predicted mask when
                ``nucleus_mask_source`` is ``mixed``.
            mask_noise_probability: Probability of applying online morphology noise
                to the chosen nucleus mask.
            mask_noise_max_radius: Maximum morphology radius for mask noise.
            boundary_gap_radius: Pixel radius used to find close lobe instances and
                build separator targets between them.
        """

        self.manifest_path = Path(manifest_path).resolve()
        self.samples: list[NucleusLobeSample] = load_lobe_manifest_samples(
            self.manifest_path,
            split=split,
        )
        self.transform = transform
        self.boundary_radius = boundary_radius
        self.nucleus_mask_source = nucleus_mask_source
        self.predicted_mask_dir = Path(predicted_mask_dir).resolve() if predicted_mask_dir else None
        self.predicted_mask_probability = predicted_mask_probability
        self.mask_noise_probability = mask_noise_probability
        self.mask_noise_max_radius = mask_noise_max_radius
        self.boundary_gap_radius = boundary_gap_radius
        if self.nucleus_mask_source in {"predicted", "mixed"} and self.predicted_mask_dir is None:
            raise ValueError("predicted_mask_dir is required for predicted or mixed mask sources.")
        if not 0.0 <= self.predicted_mask_probability <= 1.0:
            raise ValueError("predicted_mask_probability must be between 0 and 1.")
        if not 0.0 <= self.mask_noise_probability <= 1.0:
            raise ValueError("mask_noise_probability must be between 0 and 1.")

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        """Return 4-channel input, 2-channel target, raw count, and image id."""

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
        nucleus_mask_path = self._choose_nucleus_mask_path(sample)
        with Image.open(nucleus_mask_path) as mask_image:
            nucleus_mask: np.ndarray | Tensor = np.asarray(mask_image.convert("L"))
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
            lobe_instance_mask = transformed["lobe_instance_mask"]

        if self.mask_noise_probability > 0.0:
            nucleus_mask = self._maybe_add_mask_noise(nucleus_mask)

        if isinstance(rgb_image, np.ndarray):
            image_tensor = _image_to_tensor(rgb_image)
        elif isinstance(rgb_image, torch.Tensor):
            image_tensor = rgb_image.float()
        else:
            raise TypeError("Transform must return image as numpy.ndarray or torch.Tensor.")

        mask_tensor = _mask_to_tensor(nucleus_mask)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0)
        target_tensor = lobe_instance_mask_to_targets(
            lobe_instance_mask,
            boundary_radius=self.boundary_radius,
            boundary_gap_radius=self.boundary_gap_radius,
        )
        segment_count = torch.tensor(sample.segment_count, dtype=torch.long)
        return input_tensor, target_tensor.float(), segment_count, sample.image_id

    def _choose_nucleus_mask_path(self, sample: NucleusLobeSample) -> Path:
        """Choose CVAT or generated predicted nucleus mask for this sample."""

        if self.nucleus_mask_source == "cvat":
            return sample.nucleus_mask_path
        if self.predicted_mask_dir is None:
            raise ValueError("predicted_mask_dir is required.")

        predicted_path = _sample_predicted_mask_path(self.predicted_mask_dir, sample)
        if self.nucleus_mask_source == "predicted":
            if not predicted_path.is_file():
                raise FileNotFoundError(f"Predicted nucleus mask is missing: {predicted_path}")
            return predicted_path

        use_predicted = torch.rand(1).item() < self.predicted_mask_probability
        if use_predicted and predicted_path.is_file():
            return predicted_path
        return sample.nucleus_mask_path

    def _maybe_add_mask_noise(self, nucleus_mask: np.ndarray | Tensor) -> np.ndarray | Tensor:
        """Apply online morphology noise to a selected nucleus mask."""

        if torch.rand(1).item() >= self.mask_noise_probability:
            return nucleus_mask
        if isinstance(nucleus_mask, torch.Tensor):
            mask = nucleus_mask.detach().cpu().numpy()
        else:
            mask = nucleus_mask
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary_mask = mask > 0
        radius = int(torch.randint(1, self.mask_noise_max_radius + 1, (1,)).item())
        footprint = disk(radius)
        operation = int(torch.randint(0, 3, (1,)).item())
        if operation == 0:
            noised = dilation(binary_mask, footprint=footprint)
        elif operation == 1:
            noised = erosion(binary_mask, footprint=footprint)
        else:
            noised = closing(binary_mask, footprint=footprint)
        return cast(np.ndarray, noised.astype(np.uint8) * 255)
