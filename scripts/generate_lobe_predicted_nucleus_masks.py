"""Generate first-stage U-Net nucleus masks for the curated lobe dataset."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import load_lobe_manifest_samples
from src.pipeline.postprocessing import postprocess_mask
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.segmentation import UNetNucleusSegmenter

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PATHS.processed_data / "nucleus_lobes" / "predicted_nucleus_masks"
DEFAULT_WEIGHTS_PATH = PATHS.models / "unet_nucleus.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate predicted nucleus masks for lobe model training."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or another torch device string.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    samples = load_lobe_manifest_samples(args.manifest, split=args.split)
    segmenter = UNetNucleusSegmenter(weights_path=args.weights_path, device_name=args.device)
    rows: list[dict[str, str]] = []

    for sample in tqdm(samples, desc="generate masks"):
        preprocessed = preprocess_image(sample.image_path)
        raw_mask = segmenter.segment(preprocessed.normalized_rgb)
        mask = postprocess_mask(raw_mask)
        output_path = args.output_dir / sample.split / f"{sample.image_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype("uint8") * 255).save(output_path)
        rows.append(
            {
                "image_id": sample.image_id,
                "split": sample.split,
                "predicted_nucleus_mask_path": to_project_relative_str(output_path),
                "source_image_path": to_project_relative_str(sample.image_path),
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "split",
                "predicted_nucleus_mask_path",
                "source_image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated masks: {len(rows)}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")
    print(f"Manifest: {to_project_relative_str(manifest_path)}")


if __name__ == "__main__":
    main()
