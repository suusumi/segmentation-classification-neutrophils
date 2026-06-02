"""Тесты загрузки набора данных Acevedo и преобразования меток."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.datasets.acevedo_dataset import AcevedoDataset, to_binary_label
from src.datasets.transforms import get_val_transforms


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    """Создает небольшое тестовое RGB-изображение."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


def test_to_binary_label() -> None:
    """Нейтрофилы должны отображаться в 1, а все остальные метки - в 0."""
    assert to_binary_label("neutrophil") == 1
    assert to_binary_label("Neutrophil") == 1
    assert to_binary_label("monocyte") == 0


def test_folder_dataset_length_and_sample_loading(tmp_path: Path) -> None:
    """Набор данных должен загружать образцы из папок классов и возвращать тензоры."""
    _create_image(tmp_path / "neutrophil" / "n1.jpg", (255, 0, 0))
    _create_image(tmp_path / "lymphocyte" / "l1.jpg", (0, 255, 0))
    _create_image(tmp_path / "lymphocyte" / "l2.jpg", (0, 0, 255))

    dataset = AcevedoDataset(data_root=tmp_path, transform=get_val_transforms())

    assert len(dataset) == 3

    image, label, image_path = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label in {0, 1}
    assert Path(image_path).is_file()


def test_csv_dataset_loading_and_binary_labels(tmp_path: Path) -> None:
    """Набор данных должен загружать CSV-аннотации и преобразовывать метки в бинарные."""
    image_a = tmp_path / "images" / "a.jpg"
    image_b = tmp_path / "images" / "b.jpg"
    _create_image(image_a, (125, 125, 125))
    _create_image(image_b, (10, 10, 10))

    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"image_path": str(image_a), "label": "neutrophil"},
            {"image_path": str(image_b), "label": "basophil"},
        ]
    ).to_csv(csv_path, index=False)

    dataset = AcevedoDataset(csv_file=csv_path, transform=get_val_transforms())

    assert len(dataset) == 2

    first_image, first_label, _ = dataset[0]
    second_image, second_label, _ = dataset[1]
    assert isinstance(first_image, torch.Tensor)
    assert isinstance(second_image, torch.Tensor)
    assert first_label == 1
    assert second_label == 0


def test_csv_dataset_raises_readable_error_for_missing_files(tmp_path: Path) -> None:
    """Набор данных должен выбрасывать понятную ошибку, когда CSV ссылается на отсутствующие файлы."""
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame(
        [
            {"image_path": "does_not_exist.jpg", "label": "neutrophil"},
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(FileNotFoundError, match="Dataset contains missing image files"):
        AcevedoDataset(csv_file=csv_path)
