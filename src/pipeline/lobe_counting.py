"""Lobe counting backends for neutrophil nuclei."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from skimage.measure import regionprops

from src.models import UNet
from src.pipeline.segment_counting import SegmentCountConfig, SegmentCountResult, count_nucleus_segments
from src.services.errors import PipelineError
from src.training.lobe_segmentation_postprocessing import postprocess_lobe_segmentation


class LobeCounter(Protocol):
    """Interface for nucleus lobe counting backends."""

    name: str

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> "LobeCountResult":
        """Return lobe masks and count statistics."""


@dataclass(frozen=True)
class LobeCountResult:
    """Pipeline-friendly lobe counting output."""

    segment_count: SegmentCountResult
    foreground_mask: np.ndarray | None = None
    boundary_mask: np.ndarray | None = None
    split_mask: np.ndarray | None = None


@dataclass
class WatershedLobeCounter:
    """Classical fallback that counts lobes from the binary nucleus mask."""

    config: SegmentCountConfig = field(default_factory=SegmentCountConfig)
    name: str = "watershed"

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> LobeCountResult:
        """Count lobe-like components with distance-transform watershed."""

        del rgb_image
        segment_count = count_nucleus_segments(nucleus_mask, config=self.config)
        return LobeCountResult(segment_count=segment_count, split_mask=segment_count.labeled_mask)


@dataclass
class UNetLobeBoundaryCounter:
    """U-Net adapter that predicts lobe foreground and separating boundaries."""

    weights_path: Path
    name: str = "lobe_boundary_unet"
    foreground_threshold: float = 0.5
    boundary_threshold: float = 0.5
    min_segment_area_px: int = 32
    device_name: str = "auto"
    _model: UNet | None = field(default=None, init=False, repr=False)
    _image_size: int = field(default=256, init=False, repr=False)
    _mean: tuple[float, float, float] = field(
        default=(0.485, 0.456, 0.406),
        init=False,
        repr=False,
    )
    _std: tuple[float, float, float] = field(
        default=(0.229, 0.224, 0.225),
        init=False,
        repr=False,
    )

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> LobeCountResult:
        """Run lobe boundary U-Net and return countable split components."""

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PipelineError("Expected RGB image for lobe boundary U-Net.")
        if nucleus_mask.shape != rgb_image.shape[:2]:
            raise PipelineError("Nucleus mask shape must match the RGB image shape.")
        if not self.weights_path.is_file():
            raise PipelineError(
                "Lobe boundary U-Net weights are not available yet. "
                "Place weights in models/lobe_boundary_unet.pt or use watershed counting."
            )

        model = self._load_model()
        device = self._resolve_device()
        height, width = rgb_image.shape[:2]
        input_tensor = self._prepare_tensor(rgb_image, nucleus_mask).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
            probabilities = torch_functional.interpolate(
                probabilities,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        channels = probabilities.squeeze(0).cpu().numpy()
        prediction = postprocess_lobe_segmentation(
            foreground_probability=channels[0],
            boundary_probability=channels[1],
            foreground_threshold=self.foreground_threshold,
            boundary_threshold=self.boundary_threshold,
            min_segment_area_px=self.min_segment_area_px,
            support_mask=nucleus_mask,
        )
        return LobeCountResult(
            segment_count=_segment_count_from_labeled_mask(prediction.split_mask),
            foreground_mask=prediction.foreground_mask,
            boundary_mask=prediction.boundary_mask,
            split_mask=prediction.split_mask,
        )

    def _resolve_device(self) -> torch.device:
        """Return the torch device used for inference."""

        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def _load_model(self) -> UNet:
        """Load and cache the lobe boundary U-Net model."""

        if self._model is not None:
            return self._model

        device = self._resolve_device()
        try:
            checkpoint = torch.load(self.weights_path, map_location=device)
        except Exception as error:
            raise PipelineError(f"Could not load lobe boundary weights: {self.weights_path}") from error

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise PipelineError(f"Unsupported lobe boundary checkpoint: {self.weights_path}")

        model_config = checkpoint.get("model_config", {})
        base_channels = int(model_config.get("base_channels", 32))
        in_channels = int(model_config.get("in_channels", 4))
        out_channels = int(model_config.get("out_channels", 2))
        self._image_size = int(model_config.get("image_size", 256))
        self._load_normalization(checkpoint)

        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        self._model = model
        return model

    def _load_normalization(self, checkpoint: dict[str, Any]) -> None:
        """Load normalization parameters from a checkpoint when present."""

        normalization = checkpoint.get("normalization", {})
        mean = normalization.get("mean")
        std = normalization.get("std")
        if mean is not None and len(mean) == 3:
            self._mean = cast(tuple[float, float, float], tuple(float(value) for value in mean))
        if std is not None and len(std) == 3:
            self._std = cast(tuple[float, float, float], tuple(float(value) for value in std))

    def _prepare_tensor(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and concatenate RGB image with the nucleus mask."""

        image = rgb_image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = torch_functional.interpolate(
            image_tensor,
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self._mean, dtype=image_tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._std, dtype=image_tensor.dtype).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        mask_tensor = torch.from_numpy(nucleus_mask.astype(np.float32)).view(
            1,
            1,
            nucleus_mask.shape[0],
            nucleus_mask.shape[1],
        )
        mask_tensor = torch_functional.interpolate(
            mask_tensor,
            size=(self._image_size, self._image_size),
            mode="nearest",
        )
        return torch.cat([image_tensor, mask_tensor], dim=1)


def _segment_count_from_labeled_mask(labeled_mask: np.ndarray) -> SegmentCountResult:
    """Build count statistics from a labeled lobe mask."""

    regions = regionprops(labeled_mask)
    if not regions:
        return SegmentCountResult(
            segment_count=0,
            labeled_mask=np.zeros(labeled_mask.shape, dtype=np.int32),
            segment_area_mean_px=0.0,
            segment_area_min_px=0,
            segment_area_max_px=0,
        )

    areas = [int(region.area) for region in regions]
    return SegmentCountResult(
        segment_count=len(areas),
        labeled_mask=labeled_mask.astype(np.int32),
        segment_area_mean_px=float(np.mean(areas)),
        segment_area_min_px=min(areas),
        segment_area_max_px=max(areas),
    )
