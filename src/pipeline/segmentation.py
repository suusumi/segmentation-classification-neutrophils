"""Nucleus segmentation interfaces and baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu

from src.services.errors import PipelineError


class NucleusSegmenter(Protocol):
    """Interface for nucleus segmentation backends."""

    name: str

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Return a binary nucleus mask."""


@dataclass
class ThresholdNucleusSegmenter:
    """Classical fallback segmenter used until trained U-Net weights exist."""

    name: str = "threshold"
    gaussian_sigma: float = 1.0

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Segment dark blue-purple nucleus regions with adaptive thresholding."""

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PipelineError("Expected RGB image for segmentation.")

        image = rgb_image.astype(np.float32) / 255.0
        gray = rgb2gray(image)
        blue_dominance = image[:, :, 2] - image[:, :, 0]
        nucleus_score = (1.0 - gray) + np.clip(blue_dominance, 0.0, 1.0)
        smoothed = gaussian(nucleus_score, sigma=self.gaussian_sigma, preserve_range=True)

        try:
            threshold = threshold_otsu(smoothed)
        except ValueError as error:
            raise PipelineError("Could not compute a nucleus segmentation threshold.") from error

        return cast(np.ndarray, smoothed > threshold)


@dataclass
class UNetNucleusSegmenter:
    """U-Net segmenter adapter.

    The class defines the extension point for trained model inference. The
    project ships with the threshold segmenter as a deterministic baseline
    because no trained U-Net weights are currently present.
    """

    weights_path: Path
    name: str = "unet"

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Run U-Net inference when weights are available."""

        if not self.weights_path.is_file():
            raise PipelineError(
                "U-Net weights are not available yet. "
                "Use the threshold segmenter or place weights in models/unet_nucleus.pt."
            )
        raise PipelineError(
            "U-Net inference adapter is prepared but not wired to trained weights yet."
        )
