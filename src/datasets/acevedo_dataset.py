"""Определения наборов данных для классификации изображений отдельных клеток крови."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


ImageTransform = Callable[..., dict[str, Any]]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSample:
    """Одна запись набора данных с разрешенным путем изображения и метками."""
    image_path: Path
    label: str
    binary_label: int


def to_binary_label(label: str) -> int:
    """Преобразует имя класса клетки крови в метку "нейтрофил против остальных".

    Args:
        label: Исходное имя класса.

    Returns:
        ``1`` для нейтрофила и ``0`` для любого другого класса.
    """
    return int(label.strip().lower() == "neutrophil")


def _is_image_file(path: Path) -> bool:
    """Возвращает ``True``, если путь соответствует поддерживаемому расширению изображения"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _validate_missing_files(samples: Sequence[DatasetSample]) -> None:
    """Выбрасывает ошибку, если указанные файлы изображений отсутствуют"""
    missing_paths = [sample.image_path for sample in samples if not sample.image_path.is_file()]
    if not missing_paths:
        return

    preview = "\n".join(f"- {path}" for path in missing_paths[:5])
    extra_count = len(missing_paths) - min(len(missing_paths), 5)
    suffix = f"\n... and {extra_count} more." if extra_count > 0 else ""
    raise FileNotFoundError(
        "Dataset contains missing image files. First missing paths:\n"
        f"{preview}{suffix}"
    )


def _load_samples_from_directory(data_root: Path) -> list[DatasetSample]:
    """Загружает образцы набора данных из структуры каталогов с отдельной папкой на класс"""
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Expected a dataset directory, got: {data_root}")

    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories were found in dataset directory: {data_root}"
        )

    samples: list[DatasetSample] = []
    for class_dir in class_dirs:
        image_paths = sorted(path for path in class_dir.rglob("*") if _is_image_file(path))
        if not image_paths:
            continue

        label = class_dir.name
        for image_path in image_paths:
            samples.append(
                DatasetSample(
                    image_path=image_path.resolve(),
                    label=label,
                    binary_label=to_binary_label(label),
                )
            )

    if not samples:
        raise ValueError(
            f"No image files were found in dataset directory: {data_root}"
        )

    _validate_missing_files(samples)
    return samples


def _load_samples_from_csv(csv_file: Path, data_root: Path | None = None) -> list[DatasetSample]:
    """Загружает образцы набора данных из CSV-файла со столбцами ``image_path`` и ``label``."""
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset CSV file does not exist: {csv_file}")
    if not csv_file.is_file():
        raise FileNotFoundError(f"Expected a CSV file, got: {csv_file}")

    dataframe = pd.read_csv(csv_file)
    required_columns = {"image_path", "label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset CSV must contain columns {sorted(required_columns)}. Missing: {missing}"
        )

    base_dir = data_root.resolve() if data_root is not None else csv_file.parent.resolve()
    samples: list[DatasetSample] = []
    for row in dataframe.itertuples(index=False):
        raw_path = Path(str(row.image_path))
        image_path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (base_dir / raw_path).resolve()
        )
        label = str(row.label)
        samples.append(
            DatasetSample(
                image_path=image_path,
                label=label,
                binary_label=to_binary_label(label),
            )
        )

    if not samples:
        raise ValueError(f"Dataset CSV is empty: {csv_file}")

    _validate_missing_files(samples)
    return samples


def _default_image_to_tensor(image: np.ndarray) -> Tensor:
    """Преобразует RGB-изображение numpy в float-тензор с расположением каналов первыми."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image array with shape (height, width, 3)."
        )

    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    return tensor


class AcevedoDataset(Dataset[tuple[Tensor, int, str]]):
    """Набор данных PyTorch для классификации изображений на нейтрофилы и не-нейтрофилы.
    два формата входа:
    1. Структура с отдельной папкой на класс, например ``data_root/neutrophil/*.jpg``.
    2. CSV-файл со столбцами ``image_path`` и ``label``.

    Метки преобразуются в бинарную цель, где нейтрофил имеет значение ``1``,
    а любой другой класс - ``0``.
    """
    def __init__(
        self,
        data_root: str | Path | None = None,
        csv_file: str | Path | None = None,
        transform: ImageTransform | None = None,
    ) -> None:
        """Инициализирует набор данных.

        Args:
            data_root: Корневой каталог для наборов данных с отдельной папкой на класс
                или базовый каталог для разрешения относительных путей из ``csv_file``.
            csv_file: Необязательный CSV-файл аннотаций со столбцами
                ``image_path`` и ``label``.
            transform: Необязательное преобразование в стиле albumentations, которое
                принимает ``image=...`` и возвращает словарь с ключом ``image``.

        Raises:
            ValueError: Если не передан ни ``data_root``, ни ``csv_file``.
            FileNotFoundError: Если отсутствуют обязательные файлы.
        """
        if data_root is None and csv_file is None:
            raise ValueError("Provide either 'data_root' or 'csv_file' to build the dataset.")

        self.transform = transform
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.csv_file = Path(csv_file).resolve() if csv_file is not None else None

        if self.csv_file is not None:
            self.samples = _load_samples_from_csv(self.csv_file, data_root=self.data_root)
        else:
            assert self.data_root is not None
            self.samples = _load_samples_from_directory(self.data_root)

    def __len__(self) -> int:
        """Возвращает количество образцов в наборе данных."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Возвращает преобразованный тензор изображения, бинарную метку и путь изображения."""
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = np.array(image.convert("RGB"))

        if self.transform is None:
            image_tensor = _default_image_to_tensor(rgb_image)
        else:
            transformed = self.transform(image=rgb_image)
            if "image" not in transformed:
                raise KeyError("Transform output must include an 'image' entry.")

            image_tensor = transformed["image"]
            if isinstance(image_tensor, np.ndarray):
                image_tensor = _default_image_to_tensor(image_tensor)
            elif not isinstance(image_tensor, torch.Tensor):
                raise TypeError(
                    "Transform must return a torch.Tensor or numpy.ndarray in the 'image' field."
                )

        return image_tensor, sample.binary_label, str(sample.image_path)
