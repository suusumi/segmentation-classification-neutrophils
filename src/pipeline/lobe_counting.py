"""Бэкенды подсчета долей ядер нейтрофилов."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from skimage.measure import regionprops

from src.models.unet import UNet
from src.pipeline.lobe_segmentation_postprocessing import postprocess_lobe_segmentation
from src.pipeline.segment_counting import SegmentCountConfig, SegmentCountResult, count_nucleus_segments
from src.services.errors import PipelineError


class LobeCounter(Protocol):
    """Интерфейс для бэкендов подсчета долей ядра."""
    name: str

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> "LobeCountResult":
        """Возвращает маски долей и статистику подсчета."""

@dataclass(frozen=True)
class LobeCountResult:
    """Результат подсчета долей, удобный для пайплайна."""
    segment_count: SegmentCountResult
    foreground_mask: np.ndarray | None = None
    boundary_mask: np.ndarray | None = None
    split_mask: np.ndarray | None = None


@dataclass
class WatershedLobeCounter:
    """Классическая резервная реализация, подсчитывающая доли по бинарной маске ядра."""
    config: SegmentCountConfig = field(default_factory=SegmentCountConfig)
    name: str = "watershed"

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> LobeCountResult:
        """Подсчитывает компоненты, похожие на доли, с помощью watershed по преобразованию расстояния."""
        del rgb_image
        segment_count = count_nucleus_segments(nucleus_mask, config=self.config)
        return LobeCountResult(segment_count=segment_count, split_mask=segment_count.labeled_mask)


@dataclass
class UNetLobeBoundaryCounter:
    """Адаптер U-Net, предсказывающий передний план долей и разделяющие границы."""
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
        """Запускает U-Net границ долей и возвращает подсчитываемые разделенные компоненты."""
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
        """Возвращает устройство torch, используемое для инференса."""
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def _load_model(self) -> UNet:
        """Загружает и кэширует модель U-Net границ долей."""
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
        """Загружает параметры нормализации из чекпоинта, если они есть."""
        normalization = checkpoint.get("normalization", {})
        mean = normalization.get("mean")
        std = normalization.get("std")
        if mean is not None and len(mean) == 3:
            self._mean = cast(tuple[float, float, float], tuple(float(value) for value in mean))
        if std is not None and len(std) == 3:
            self._std = cast(tuple[float, float, float], tuple(float(value) for value in std))

    def _prepare_tensor(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> torch.Tensor:
        """Изменяет размер, нормализует и объединяет RGB-изображение с маской ядра."""
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


@dataclass
class YOLOLobeInstanceCounter:
    """Адаптер Ultralytics YOLO-seg, считающий одну предсказанную маску на долю ядра."""
    weights_path: Path
    name: str = "yolo_lobes_seg"
    confidence: float = 0.40
    iou: float = 0.30
    image_size: int = 640
    min_mask_area_px: int = 16
    device_name: str = "auto"
    _model: Any | None = field(default=None, init=False, repr=False)

    def count(self, rgb_image: np.ndarray, nucleus_mask: np.ndarray) -> LobeCountResult:
        """Запускает сегментацию экземпляров долей YOLO и возвращает подсчитываемые компоненты."""
        del nucleus_mask
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PipelineError("Expected RGB image for YOLO lobe segmentation.")
        if not self.weights_path.is_file():
            raise PipelineError(
                "YOLO lobe weights are not available yet. "
                "Train models/yolo_lobes_seg.pt or set YOLO_LOBE_WEIGHTS_PATH."
            )

        model = self._load_model()
        predict_kwargs: dict[str, Any] = {
            "source": Image.fromarray(rgb_image.astype(np.uint8)),
            "imgsz": self.image_size,
            "conf": self.confidence,
            "iou": self.iou,
            "stream": False,
            "verbose": False,
        }
        device = self._device_argument()
        if device is not None:
            predict_kwargs["device"] = device

        try:
            results = list(model.predict(**predict_kwargs))
        except Exception as error:
            raise PipelineError("YOLO lobe inference failed.") from error
        if not results:
            raise PipelineError("YOLO lobe inference returned no results.")

        split_mask = self._result_to_labeled_mask(results[0], shape=rgb_image.shape[:2])
        return LobeCountResult(
            segment_count=_segment_count_from_labeled_mask(split_mask),
            foreground_mask=split_mask > 0,
            split_mask=split_mask,
        )

    def _load_model(self) -> Any:
        """Загружает и кэширует модель Ultralytics."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise PipelineError(
                "Ultralytics is not installed. Install it with "
                ".\\.venv\\Scripts\\python.exe -m pip install ultralytics"
            ) from error

        try:
            self._model = YOLO(self.weights_path.as_posix())
        except Exception as error:
            raise PipelineError(f"Could not load YOLO lobe weights: {self.weights_path}") from error
        return self._model

    def _device_argument(self) -> str | None:
        """Возвращает аргумент устройства для Ultralytics."""
        if self.device_name == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return self.device_name

    def _result_to_labeled_mask(self, result: Any, shape: tuple[int, int]) -> np.ndarray:
        """Преобразует маски Ultralytics в изображение размеченных компонентов."""
        height, width = shape
        labeled = np.zeros((height, width), dtype=np.int32)
        masks = getattr(result, "masks", None)
        data = getattr(masks, "data", None) if masks is not None else None
        if data is None:
            return labeled

        mask_array = data.detach().cpu().numpy()
        if mask_array.ndim != 3:
            return labeled

        boxes = getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None) if boxes is not None else None
        if confidences is None:
            confidence_values = np.ones(mask_array.shape[0], dtype=np.float32)
        else:
            confidence_values = confidences.detach().cpu().numpy().astype(np.float32)

        instances: list[tuple[float, np.ndarray]] = []
        for index, raw_mask in enumerate(mask_array):
            mask = self._resize_mask(raw_mask > 0.5, size=(width, height))
            area = int(mask.sum())
            if area < self.min_mask_area_px:
                continue
            confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
            instances.append((confidence, mask))

        for label_value, (_, mask) in enumerate(
            sorted(instances, key=lambda item: item[0], reverse=True),
            start=1,
        ):
            labeled[np.logical_and(mask, labeled == 0)] = label_value
        return labeled

    @staticmethod
    def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Изменяет размер бинарной маски до ``(width, height)`` методом ближайшего соседа."""
        if mask.shape == (size[1], size[0]):
            return mask.astype(bool)
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        return np.asarray(image.resize(size, resample=Image.Resampling.NEAREST)) > 0


def _segment_count_from_labeled_mask(labeled_mask: np.ndarray) -> SegmentCountResult:
    """Строит статистику подсчета по размеченной маске долей."""
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
