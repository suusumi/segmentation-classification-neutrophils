"""Проверка и предобработка изображений одиночных нейтрофилов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import exposure

from src.core.settings import SETTINGS
from src.services.errors import UploadValidationError


@dataclass(frozen=True)
class PreprocessedImage:
    """RGB-изображение и нормализованное представление для последующих шагов."""
    rgb: np.ndarray
    normalized_rgb: np.ndarray
    width: int
    height: int


def validate_image_path(path: Path) -> None:
    """Проверяет расширение файла изображения и его существование."""
    if not path.is_file():
        raise UploadValidationError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in SETTINGS.allowed_image_extensions:
        raise UploadValidationError(f"Unsupported image extension: {path.suffix}")


def load_rgb_image(path: Path) -> np.ndarray:
    """Загружает изображение как RGB-массив numpy."""
    validate_image_path(path)
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except UnidentifiedImageError as error:
        raise UploadValidationError("Uploaded file is not a readable image.") from error


def preprocess_image(path: Path) -> PreprocessedImage:
    """Загружает и нормализует изображение для сегментации ядра."""
    rgb = load_rgb_image(path)
    normalized = exposure.rescale_intensity(rgb, in_range="image", out_range=(0, 255)).astype(
        np.uint8
    )
    height, width = normalized.shape[:2]
    return PreprocessedImage(rgb=rgb, normalized_rgb=normalized, width=width, height=height)

