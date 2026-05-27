"""Convert CVAT Segmentation Mask exports into a U-Net training dataset."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CVAT_DIR = PATHS.data / "cvat"
DEFAULT_SOURCE_IMAGES_DIR = PATHS.raw_data / "acevedo" / "neutrophil"
DEFAULT_OUTPUT_DIR = PATHS.processed_data / "nucleus_segmentation" / "curated"


@dataclass(frozen=True)
class CuratedNucleusRow:
    """One curated segmentation sample."""

    image_id: str
    source_image_path: Path
    image_path: Path
    mask_path: Path
    split: str

    def to_row(self) -> dict[str, str]:
        """Return a manifest row with project-relative paths."""

        return {
            "image_id": self.image_id,
            "source_image_path": to_project_relative_str(self.source_image_path),
            "image_path": to_project_relative_str(self.image_path),
            "mask_path": to_project_relative_str(self.mask_path),
            "split": self.split,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Convert a CVAT Segmentation Mask export into image/mask pairs."
    )
    parser.add_argument(
        "--cvat-dir",
        type=Path,
        default=DEFAULT_CVAT_DIR,
        help="Unpacked CVAT export directory containing SegmentationClass.",
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=DEFAULT_SOURCE_IMAGES_DIR,
        help="Directory with source neutrophil images matched by filename stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Curated dataset output directory.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing curated image and mask files.",
    )
    return parser.parse_args()


def _is_image_file(path: Path) -> bool:
    """Return True for supported image files."""

    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _load_annotation_ids(cvat_dir: Path) -> list[str]:
    """Load ordered image ids from the CVAT export when possible."""

    default_set = cvat_dir / "ImageSets" / "Segmentation" / "default.txt"
    segmentation_class_dir = cvat_dir / "SegmentationClass"
    if default_set.is_file():
        ids = [line.strip() for line in default_set.read_text(encoding="utf-8").splitlines()]
        return [image_id for image_id in ids if image_id]
    return sorted(path.stem for path in segmentation_class_dir.glob("*.png"))


def _index_source_images(source_images_dir: Path) -> dict[str, Path]:
    """Index source images by filename stem."""

    if not source_images_dir.is_dir():
        raise NotADirectoryError(f"Source images directory does not exist: {source_images_dir}")
    return {
        path.stem: path.resolve()
        for path in sorted(source_images_dir.rglob("*"))
        if _is_image_file(path)
    }


def _validate_fractions(val_fraction: float, test_fraction: float) -> None:
    """Validate split fractions."""

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
    """Assign deterministic train/val/test splits."""

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


def load_cvat_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a CVAT SegmentationClass PNG as a binary uint8 mask."""

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 2:
        binary = mask > 0
    elif mask.ndim == 3:
        binary = np.any(mask[:, :, :3] > 0, axis=2)
    else:
        raise ValueError(f"Unsupported mask shape for {mask_path}: {mask.shape}")

    if not binary.any():
        raise ValueError(f"Mask is empty: {mask_path}")
    return binary.astype(np.uint8) * 255


def _copy_image(source_path: Path, destination_path: Path, overwrite: bool) -> None:
    """Copy one source image into the curated dataset."""

    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _save_mask(mask: np.ndarray, destination_path: Path, overwrite: bool) -> None:
    """Save one binary mask into the curated dataset."""

    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(destination_path)


def convert_cvat_export(
    cvat_dir: Path,
    source_images_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> list[CuratedNucleusRow]:
    """Convert CVAT masks into a curated image/mask manifest."""

    segmentation_class_dir = cvat_dir / "SegmentationClass"
    if not segmentation_class_dir.is_dir():
        raise NotADirectoryError(
            f"CVAT SegmentationClass directory not found: {segmentation_class_dir}"
        )

    image_ids = _load_annotation_ids(cvat_dir)
    if not image_ids:
        raise ValueError(f"No CVAT annotation ids found in: {cvat_dir}")

    source_images = _index_source_images(source_images_dir)
    missing_images = [image_id for image_id in image_ids if image_id not in source_images]
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(f"Missing source images for annotation ids: {preview}")

    split_by_id = assign_splits(
        image_ids=image_ids,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    rows: list[CuratedNucleusRow] = []
    for image_id in sorted(image_ids):
        source_image_path = source_images[image_id]
        split = split_by_id[image_id]
        image_output_path = output_dir / "images" / split / source_image_path.name
        mask_output_path = output_dir / "masks" / split / f"{image_id}.png"
        source_mask_path = segmentation_class_dir / f"{image_id}.png"

        mask = load_cvat_binary_mask(source_mask_path)
        with Image.open(source_mask_path) as source_mask:
            source_mask_size = source_mask.size
        with Image.open(source_image_path) as source_image:
            if source_image.size != source_mask_size:
                raise ValueError(
                    "Image and mask sizes differ for "
                    f"{image_id}: image={source_image.size}, mask={source_mask_size}"
                )

        _copy_image(source_image_path, image_output_path, overwrite=overwrite)
        _save_mask(mask, mask_output_path, overwrite=overwrite)
        rows.append(
            CuratedNucleusRow(
                image_id=image_id,
                source_image_path=source_image_path,
                image_path=image_output_path,
                mask_path=mask_output_path,
                split=split,
            )
        )

    save_manifest(rows, output_dir / "manifest.csv")
    return rows


def save_manifest(rows: list[CuratedNucleusRow], manifest_path: Path) -> Path:
    """Write a curated dataset manifest."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "source_image_path", "image_path", "mask_path", "split"]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return manifest_path


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    rows = convert_cvat_export(
        cvat_dir=args.cvat_dir,
        source_images_dir=args.source_images_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    split_counts = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "val", "test")
    }
    print(f"Converted CVAT samples: {len(rows)}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print(f"Manifest: {to_project_relative_str(args.output_dir / 'manifest.csv')}")
    print(
        "Splits: "
        f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
    )


if __name__ == "__main__":
    main()
