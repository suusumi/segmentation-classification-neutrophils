"""Интерфейсы сегментации ядра и базовые реализации."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu

from src.models.unet import UNet
from src.services.errors import PipelineError


class NucleusSegmenter(Protocol):
    """Интерфейс для бэкендов сегментации ядра."""
    name: str

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Возвращает бинарную маску ядра."""

@dataclass
class ThresholdNucleusSegmenter:
    """Классический резервный сегментатор, используемый до появления обученных весов U-Net."""
    name: str = "threshold"
    gaussian_sigma: float = 1.0

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Сегментирует темные сине-фиолетовые области ядра с помощью адаптивной пороговой обработки."""
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
    """Адаптер сегментатора U-Net для обученных весов сегментации ядра."""
    weights_path: Path
    name: str = "unet"
    threshold: float = 0.5
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

    def segment(self, rgb_image: np.ndarray) -> np.ndarray:
        """Запускает инференс U-Net и возвращает бинарную маску ядра."""
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PipelineError("Expected RGB image for U-Net segmentation.")
        if not self.weights_path.is_file():
            raise PipelineError(
                "U-Net weights are not available yet. "
                "Use the threshold segmenter or place weights in models/unet_nucleus.pt."
            )

        model = self._load_model()
        device = self._resolve_device()
        height, width = rgb_image.shape[:2]
        input_tensor = self._prepare_tensor(rgb_image).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
            probabilities = torch_functional.interpolate(
                probabilities,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        mask = probabilities.squeeze().cpu().numpy() > self.threshold
        return cast(np.ndarray, mask)

    def _resolve_device(self) -> torch.device:
        """Возвращает устройство torch, используемое для инференса."""
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def _load_model(self) -> UNet:
        """Загружает и кэширует модель U-Net."""
        if self._model is not None:
            return self._model

        device = self._resolve_device()
        try:
            checkpoint = torch.load(self.weights_path, map_location=device)
        except Exception as error:
            raise PipelineError(f"Could not load U-Net weights: {self.weights_path}") from error

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_config = checkpoint.get("model_config", {})
            base_channels = int(model_config.get("base_channels", 32))
            self._image_size = int(model_config.get("image_size", 256))
            self._load_normalization(checkpoint)
            state_dict = checkpoint["model_state_dict"]
        else:
            base_channels = 32
            state_dict = checkpoint

        model = UNet(base_channels=base_channels)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._model = model
        return model

    def _load_normalization(self, checkpoint: dict[str, Any]) -> None:
        """Загружает параметры нормализации из чекпоинта, если они есть."""
        normalization = checkpoint.get("normalization", {})
        mean = normalization.get("mean")
        std = normalization.get("std")
        if mean is not None and len(mean) == 3:
            self._mean = cast(tuple[float, float, float], tuple(float(value) for value in mean))
        if std is not None and len(std) == 3:
            self._std = cast(tuple[float, float, float], tuple(float(value) for value in std))

    def _prepare_tensor(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Изменяет размер и нормализует RGB-изображение для инференса U-Net."""
        image = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = torch_functional.interpolate(
            tensor,
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self._mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._std, dtype=tensor.dtype).view(1, 3, 1, 1)
        return (tensor - mean) / std
