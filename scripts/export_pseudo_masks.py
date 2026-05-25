"""Generate pseudo-label nucleus masks for manual annotation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.pipeline.artifacts import save_mask, save_overlay
from src.pipeline.postprocessing import postprocess_mask
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.segmentation import ThresholdNucleusSegmenter

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_OUTPUT_DIR = PATHS.processed_data / "nucleus_segmentation" / "pseudo_labels"


@dataclass(frozen=True)
class PseudoMaskRecord:
    """Manifest row for one generated pseudo-label."""

    image_id: str
    source_image_path: str
    image_path: str
    pseudo_mask_path: str
    overlay_path: str
    split: str
    annotation_status: str = "pseudo"
    needs_review: bool = True

    def to_row(self) -> dict[str, str]:
        """Return a CSV-friendly row."""

        return {
            "image_id": self.image_id,
            "source_image_path": self.source_image_path,
            "image_path": self.image_path,
            "pseudo_mask_path": self.pseudo_mask_path,
            "overlay_path": self.overlay_path,
            "split": self.split,
            "annotation_status": self.annotation_status,
            "needs_review": str(self.needs_review).lower(),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo-label nucleus masks from Acevedo neutrophil images "
            "using the current threshold segmentation baseline."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PATHS.raw_data / "acevedo" / "neutrophil",
        help="Directory with source neutrophil images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where images, pseudo masks, overlays, and manifest.csv are saved.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PATHS.processed_data / "acevedo_splits",
        help="Optional directory with train.csv, val.csv, and test.csv split files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks and overlays even when output files already exist.",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Do not save visual overlay PNG files.",
    )
    return parser.parse_args()


def _is_image_file(path: Path) -> bool:
    """Return True when path is a supported image file."""

    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_image_paths(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Collect image paths from a source directory."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected an image directory, got: {input_dir}")

    image_paths = sorted(path for path in input_dir.rglob("*") if _is_image_file(path))
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No supported image files found in: {input_dir}")
    return image_paths


def _resolve_manifest_path(path_value: str) -> Path:
    """Resolve a split CSV path value relative to the project root when needed."""

    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_split_map(splits_dir: Path) -> dict[Path, str]:
    """Load project split CSVs into a source-image-to-split map."""

    split_map: dict[Path, str] = {}
    for split_name in ("train", "val", "test"):
        split_path = splits_dir / f"{split_name}.csv"
        if not split_path.is_file():
            continue
        with split_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path_value = row.get("image_path")
                if not image_path_value:
                    continue
                split_map[_resolve_manifest_path(image_path_value)] = split_name
    return split_map


def _prepare_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create and return images, masks, and overlays directories."""

    images_dir = output_dir / "images"
    masks_dir = output_dir / "pseudo_masks"
    overlays_dir = output_dir / "overlays"
    for directory in (images_dir, masks_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, overlays_dir


def _copy_source_image(source_path: Path, destination_path: Path, overwrite: bool) -> None:
    """Copy a source image into the annotation dataset."""

    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def generate_pseudo_masks(
    image_paths: list[Path],
    output_dir: Path,
    split_map: dict[Path, str] | None = None,
    overwrite: bool = False,
    save_overlays: bool = True,
) -> list[PseudoMaskRecord]:
    """Generate pseudo-label masks and return manifest records."""

    images_dir, masks_dir, overlays_dir = _prepare_output_dirs(output_dir)
    segmenter = ThresholdNucleusSegmenter()
    records: list[PseudoMaskRecord] = []
    split_map = split_map or {}

    for index, source_path in enumerate(image_paths, start=1):
        source_path = source_path.resolve()
        image_id = source_path.stem
        image_output_path = images_dir / source_path.name
        mask_output_path = masks_dir / f"{image_id}.png"
        overlay_output_path = overlays_dir / f"{image_id}_overlay.png"

        _copy_source_image(source_path, image_output_path, overwrite=overwrite)

        if overwrite or not mask_output_path.exists() or (
            save_overlays and not overlay_output_path.exists()
        ):
            preprocessed = preprocess_image(source_path)
            raw_mask = segmenter.segment(preprocessed.normalized_rgb)
            mask = postprocess_mask(raw_mask)
            save_mask(mask, mask_output_path)
            if save_overlays:
                save_overlay(preprocessed.rgb, mask, overlay_output_path)

        records.append(
            PseudoMaskRecord(
                image_id=image_id,
                source_image_path=to_project_relative_str(source_path),
                image_path=to_project_relative_str(image_output_path),
                pseudo_mask_path=to_project_relative_str(mask_output_path),
                overlay_path=to_project_relative_str(overlay_output_path)
                if save_overlays
                else "",
                split=split_map.get(source_path, "unsplit"),
            )
        )
        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    return records


def save_manifest(records: list[PseudoMaskRecord], output_dir: Path) -> Path:
    """Save manifest.csv for the generated pseudo-label dataset."""

    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PseudoMaskRecord("", "", "", "", "", "").to_row().keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return manifest_path


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    image_paths = collect_image_paths(args.input_dir, limit=args.limit)
    split_map = load_split_map(args.splits_dir)
    records = generate_pseudo_masks(
        image_paths=image_paths,
        output_dir=args.output_dir,
        split_map=split_map,
        overwrite=args.overwrite,
        save_overlays=not args.no_overlays,
    )
    manifest_path = save_manifest(records, args.output_dir)

    print(f"Generated pseudo masks: {len(records)}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print(f"Manifest: {to_project_relative_str(manifest_path)}")


if __name__ == "__main__":
    main()
