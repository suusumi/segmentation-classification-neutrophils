"""Подготавливает стратифицированные CSV-разбиения для набора клеток крови Acevedo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test CSV splits for the Acevedo dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the folder-per-class dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where train.csv, val.csv, and test.csv will be saved.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of the full dataset reserved for validation.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of the full dataset reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for stratified splitting.",
    )
    parser.add_argument(
        "--relative_to",
        type=Path,
        default=PROJECT_ROOT,
        help="Base directory used to store portable relative image paths.",
    )
    return parser.parse_args()


def _is_image_file(path: Path) -> bool:
    """Возвращает ``True``, если путь указывает на поддерживаемый файл изображения."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _portable_image_path(image_path: Path, relative_to: Path) -> str:
    """Возвращает переносимый путь изображения в POSIX-стиле для вывода CSV."""
    try:
        return image_path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return image_path.resolve().as_posix()


def _scan_dataset(input_dir: Path, relative_to: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Сканирует набор данных с отдельной папкой на каждый класс в датафрейм.

    Args:
        input_dir: Каталог, содержащий по одному подкаталогу на каждый класс клеток.

    Returns:
        Датафрейм со столбцами ``image_path``, ``label`` и ``binary_label``.
    """
    from src.datasets.acevedo_dataset import to_binary_label

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected a dataset directory, got: {input_dir}")

    records: list[dict[str, str | int]] = []
    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories were found in dataset directory: {input_dir}"
        )

    for class_dir in class_dirs:
        label = class_dir.name
        image_paths = sorted(path for path in class_dir.rglob("*") if _is_image_file(path))
        for image_path in image_paths:
            records.append(
                {
                    "image_path": _portable_image_path(image_path, relative_to),
                    "label": label,
                    "binary_label": to_binary_label(label),
                }
            )

    if not records:
        raise ValueError(f"No image files were found in dataset directory: {input_dir}")

    return pd.DataFrame.from_records(records)


def _validate_split_sizes(val_size: float, test_size: float) -> None:
    """Проверяет запрошенные размеры разбиений."""
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            "val_size + test_size must be less than 1.0 to leave room for a training split."
        )


def create_splits(
    dataframe: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Создает стратифицированные датафреймы train/val/test.

    Стратификация выполняется по исходной многоклассовой метке, чтобы сохранить
    состав подтипов не-нейтрофилов в разных разбиениях.
    """
    _validate_split_sizes(val_size, test_size)

    try:
        train_val_df, test_df = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=seed,
            stratify=dataframe["label"],
        )

        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=seed,
            stratify=train_val_df["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Could not create stratified splits. Check class balance and split sizes."
        ) from error

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(
        drop=True
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Сохраняет датафреймы разбиений в CSV-файлы."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    """Точка входа CLI для подготовки разбиений набора данных Acevedo."""
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    relative_to = args.relative_to.resolve()

    dataframe = _scan_dataset(input_dir, relative_to=relative_to)
    train_df, val_df, test_df = create_splits(
        dataframe=dataframe,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    save_splits(train_df, val_df, test_df, output_dir)

    print(f"Scanned {len(dataframe)} images from: {input_dir}")
    print(f"Train split: {len(train_df)}")
    print(f"Validation split: {len(val_df)}")
    print(f"Test split: {len(test_df)}")
    print(f"Saved CSV files to: {output_dir}")


if __name__ == "__main__":
    main()
