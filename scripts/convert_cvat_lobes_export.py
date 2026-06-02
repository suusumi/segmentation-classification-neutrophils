"""Преобразует экспорт экземпляров COCO из CVAT в набор данных долей ядра."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CVAT_DIR = PATHS.data / "cvat_lobes"
DEFAULT_SOURCE_IMAGES_DIR = PATHS.raw_data / "acevedo" / "neutrophil"
DEFAULT_OUTPUT_DIR = PATHS.processed_data / "nucleus_lobes" / "curated"
DEFAULT_LABEL = "nucleus_lobe"


@dataclass(frozen=True)
class CuratedLobeRow:
    """Один курируемый образец с метками долей ядра на уровне экземпляров."""
    image_id: str
    source_image_path: Path
    image_path: Path
    nucleus_mask_path: Path
    lobe_instance_mask_path: Path
    segment_count: int
    split: str

    def to_row(self) -> dict[str, str | int]:
        """Возвращает строку манифеста с путями относительно проекта."""
        return {
            "image_id": self.image_id,
            "source_image_path": to_project_relative_str(self.source_image_path),
            "image_path": to_project_relative_str(self.image_path),
            "nucleus_mask_path": to_project_relative_str(self.nucleus_mask_path),
            "lobe_instance_mask_path": to_project_relative_str(self.lobe_instance_mask_path),
            "segment_count": self.segment_count,
            "split": self.split,
        }


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Convert CVAT COCO instance annotations into lobe instance masks."
    )
    parser.add_argument(
        "--cvat-dir",
        type=Path,
        default=DEFAULT_CVAT_DIR,
        help="Unpacked CVAT COCO export directory.",
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=None,
        help="Path to a COCO JSON file. Overrides --cvat-dir discovery.",
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=DEFAULT_SOURCE_IMAGES_DIR,
        help="Directory with source neutrophil images matched by file name or stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Curated lobe dataset output directory.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="COCO category name used for individual nucleus lobes.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing curated image and mask files.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include images without nucleus_lobe annotations as an error instead of skipping them.",
    )
    return parser.parse_args()


def _is_image_file(path: Path) -> bool:
    """Возвращает True для поддерживаемых файлов изображений."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _find_coco_json(cvat_dir: Path) -> Path:
    """Находит JSON с аннотациями COCO в распакованном экспорте CVAT."""
    candidates = [
        cvat_dir / "annotations" / "instances_default.json",
        cvat_dir / "instances_default.json",
        cvat_dir / "annotations" / "default.json",
        cvat_dir / "default.json",
    ]
    candidates.extend(sorted(cvat_dir.rglob("*.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a COCO JSON file in: {cvat_dir}")


def _load_coco(coco_json: Path) -> dict[str, Any]:
    """Загружает файл аннотаций COCO и выполняет минимальную проверку."""
    if not coco_json.is_file():
        raise FileNotFoundError(f"COCO JSON does not exist: {coco_json}")
    data = cast(dict[str, Any], json.loads(coco_json.read_text(encoding="utf-8")))
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"COCO JSON is missing required key: {key}")
    return data


def _index_source_images(source_images_dir: Path) -> dict[str, Path]:
    """Индексирует исходные изображения по имени файла и основе имени."""
    if not source_images_dir.is_dir():
        raise NotADirectoryError(f"Source images directory does not exist: {source_images_dir}")

    index: dict[str, Path] = {}
    for path in sorted(source_images_dir.rglob("*")):
        if not _is_image_file(path):
            continue
        resolved = path.resolve()
        index[path.name] = resolved
        index.setdefault(path.stem, resolved)
    return index


def _resolve_source_image(file_name: str, source_images: dict[str, Path]) -> Path:
    """Сопоставляет имя файла изображения COCO с индексом исходных изображений."""
    file_path = Path(file_name)
    for key in (file_path.as_posix(), file_path.name, file_path.stem):
        if key in source_images:
            return source_images[key]
    raise FileNotFoundError(f"Missing source image for COCO file_name: {file_name}")


def _validate_fractions(val_fraction: float, test_fraction: float) -> None:
    """Проверяет доли разбиений."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave at least one train sample.")


def assign_splits(
    image_ids: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Назначает детерминированные разбиения train/val/test."""
    _validate_fractions(val_fraction, test_fraction)
    shuffled = image_ids.copy()
    random.Random(seed).shuffle(shuffled)

    test_count = round(len(shuffled) * test_fraction)
    val_count = round(len(shuffled) * val_fraction)
    split_by_id: dict[str, str] = {}
    for index, image_id in enumerate(shuffled):
        if index < test_count:
            split_by_id[image_id] = "test"
        elif index < test_count + val_count:
            split_by_id[image_id] = "val"
        else:
            split_by_id[image_id] = "train"
    return split_by_id


def _category_ids_by_name(coco: dict[str, Any], label: str) -> set[int]:
    """Возвращает идентификаторы категорий COCO, соответствующие запрошенной метке."""
    category_ids = {
        int(category["id"])
        for category in coco["categories"]
        if str(category.get("name", "")).strip() == label
    }
    if not category_ids:
        category_names = ", ".join(
            sorted(str(category.get("name", "")) for category in coco["categories"])
        )
        raise ValueError(f"Label '{label}' was not found in COCO categories: {category_names}")
    return category_ids


def _polygon_points(segmentation: list[float]) -> list[tuple[float, float]]:
    """Преобразует один плоский полигон COCO в точки, совместимые с PIL."""
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
        raise ValueError("COCO polygon segmentation must contain x/y coordinate pairs.")
    return [
        (float(segmentation[index]), float(segmentation[index + 1]))
        for index in range(0, len(segmentation), 2)
    ]


def _draw_annotation(
    instance_mask: np.ndarray,
    annotation: dict[str, Any],
    instance_id: int,
) -> None:
    """Рисует одну полигональную аннотацию в маске экземпляров."""
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        raise ValueError(
            "COCO RLE segmentations are not supported yet. "
            "Export CVAT shapes as polygons for nucleus_lobe annotations."
        )
    if not isinstance(segmentation, list):
        raise ValueError("COCO annotation segmentation must be a polygon list.")

    object_mask = Image.new("L", (instance_mask.shape[1], instance_mask.shape[0]), 0)
    draw = ImageDraw.Draw(object_mask)
    for polygon in segmentation:
        if not polygon:
            continue
        draw.polygon(_polygon_points(polygon), fill=1)

    object_array = np.asarray(object_mask, dtype=bool)
    if not object_array.any():
        raise ValueError(f"Annotation {annotation.get('id')} produced an empty lobe mask.")
    instance_mask[object_array] = instance_id


def build_lobe_instance_mask(
    image_width: int,
    image_height: int,
    annotations: list[dict[str, Any]],
) -> np.ndarray:
    """Строит uint16-маску экземпляров, где каждая доля имеет отдельный id."""
    if not annotations:
        raise ValueError("Image has no nucleus_lobe annotations.")
    if len(annotations) > np.iinfo(np.uint16).max:
        raise ValueError("Too many lobe annotations for a uint16 instance mask.")

    instance_mask = np.zeros((image_height, image_width), dtype=np.uint16)
    for instance_id, annotation in enumerate(
        sorted(annotations, key=lambda item: int(item.get("id", 0))),
        start=1,
    ):
        _draw_annotation(instance_mask, annotation, instance_id)
    return instance_mask


def _copy_image(source_path: Path, destination_path: Path, overwrite: bool) -> None:
    """Копирует одно исходное изображение в курируемый набор данных."""
    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _save_instance_mask(mask: np.ndarray, destination_path: Path, overwrite: bool) -> None:
    """Сохраняет uint16-маску экземпляров долей."""
    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint16)).save(destination_path)


def _save_binary_mask(mask: np.ndarray, destination_path: Path, overwrite: bool) -> None:
    """Сохраняет объединение всех экземпляров долей как бинарную маску ядра."""
    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(destination_path)


def convert_cvat_lobes_export(
    cvat_dir: Path,
    source_images_dir: Path,
    output_dir: Path,
    coco_json: Path | None = None,
    label: str = DEFAULT_LABEL,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
    skip_empty: bool = True,
) -> list[CuratedLobeRow]:
    """Преобразует экспорт экземпляров COCO из CVAT в курируемые маски долей."""
    annotation_json = coco_json or _find_coco_json(cvat_dir)
    coco = _load_coco(annotation_json)
    category_ids = _category_ids_by_name(coco, label=label)
    source_images = _index_source_images(source_images_dir)

    images = {
        int(image["id"]): image
        for image in coco["images"]
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        if int(annotation.get("category_id", -1)) not in category_ids:
            continue
        image_id = int(annotation["image_id"])
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    annotated_images = {
        image_id: image
        for image_id, image in images.items()
        if annotations_by_image[image_id]
    }
    if not annotated_images:
        raise ValueError(f"No annotations with label '{label}' were found in: {annotation_json}")
    if not skip_empty:
        empty_count = len(images) - len(annotated_images)
        if empty_count:
            raise ValueError(f"COCO export contains {empty_count} images without {label} annotations.")

    image_keys = [Path(str(image["file_name"])).stem for image in annotated_images.values()]
    split_by_id = assign_splits(
        image_ids=image_keys,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    rows: list[CuratedLobeRow] = []
    for coco_image_id, image in sorted(
        annotated_images.items(),
        key=lambda item: str(item[1]["file_name"]),
    ):
        file_name = str(image["file_name"])
        image_key = Path(file_name).stem
        split = split_by_id[image_key]
        source_image_path = _resolve_source_image(file_name, source_images)
        annotations = annotations_by_image[coco_image_id]
        instance_mask = build_lobe_instance_mask(
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            annotations=annotations,
        )

        with Image.open(source_image_path) as source_image:
            if source_image.size != (int(image["width"]), int(image["height"])):
                raise ValueError(
                    "Source image and COCO dimensions differ for "
                    f"{file_name}: image={source_image.size}, "
                    f"coco={(int(image['width']), int(image['height']))}"
                )

        image_output_path = output_dir / "images" / split / source_image_path.name
        nucleus_mask_path = output_dir / "nucleus_masks" / split / f"{image_key}.png"
        lobe_instance_mask_path = output_dir / "lobe_instance_masks" / split / f"{image_key}.png"

        _copy_image(source_image_path, image_output_path, overwrite=overwrite)
        _save_binary_mask(instance_mask, nucleus_mask_path, overwrite=overwrite)
        _save_instance_mask(instance_mask, lobe_instance_mask_path, overwrite=overwrite)

        rows.append(
            CuratedLobeRow(
                image_id=image_key,
                source_image_path=source_image_path,
                image_path=image_output_path,
                nucleus_mask_path=nucleus_mask_path,
                lobe_instance_mask_path=lobe_instance_mask_path,
                segment_count=int(instance_mask.max()),
                split=split,
            )
        )

    save_manifest(rows, output_dir / "manifest.csv")
    return rows


def save_manifest(rows: list[CuratedLobeRow], manifest_path: Path) -> Path:
    """Записывает манифест курируемого набора данных долей."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "source_image_path",
        "image_path",
        "nucleus_mask_path",
        "lobe_instance_mask_path",
        "segment_count",
        "split",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return manifest_path


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    rows = convert_cvat_lobes_export(
        cvat_dir=args.cvat_dir,
        source_images_dir=args.source_images_dir,
        output_dir=args.output_dir,
        coco_json=args.coco_json,
        label=args.label,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_empty=not args.include_empty,
    )
    split_counts = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "val", "test")
    }
    print(f"Converted lobe samples: {len(rows)}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print(f"Manifest: {to_project_relative_str(args.output_dir / 'manifest.csv')}")
    print(
        "Splits: "
        f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
    )


if __name__ == "__main__":
    main()
