"""Evaluate a trained nucleus lobe count baseline."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import (
    NucleusLobeCountDataset,
    load_lobe_count_values,
)
from src.datasets.transforms import get_lobe_count_val_transforms
from src.models import LobeCountCNN
from src.training.lobe_count_metrics import compute_lobe_count_metrics

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_WEIGHTS_PATH = PATHS.models / "lobe_count_baseline.pt"
DEFAULT_OUTPUT_DIR = PATHS.outputs / "lobe_count_eval"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a nucleus lobe count classifier.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hypersegmentation-threshold", type=int, default=5)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or another torch device string.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Resolve a CLI device name into a torch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_checkpoint(weights_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a lobe count checkpoint."""

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported lobe count checkpoint format: {weights_path}")
    return cast(dict[str, Any], checkpoint)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[LobeCountCNN, list[int], int]:
    """Build a model and class list from a checkpoint."""

    model_config = checkpoint.get("model_config", {})
    class_values = [int(value) for value in checkpoint.get("class_values", [])]
    if not class_values:
        raise ValueError("Checkpoint is missing class_values.")

    model = LobeCountCNN(
        in_channels=int(model_config.get("in_channels", 4)),
        num_classes=int(model_config.get("num_classes", len(class_values))),
        base_channels=int(model_config.get("base_channels", 24)),
        dropout=float(model_config.get("dropout", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_values, int(model_config.get("image_size", 256))


@torch.no_grad()
def predict(
    model: LobeCountCNN,
    dataloader: DataLoader,
    class_values: list[int],
    device: torch.device,
    hypersegmentation_threshold: int,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Run prediction and return per-sample rows plus raw targets/predictions."""

    rows: list[dict[str, Any]] = []
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for inputs, _, segment_counts, image_ids in tqdm(dataloader, desc="evaluate"):
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predicted_indexes = probabilities.argmax(dim=1).tolist()
        target_counts = [int(value) for value in segment_counts.cpu().tolist()]

        for item_index, image_id in enumerate(image_ids):
            prediction = class_values[int(predicted_indexes[item_index])]
            confidence = float(probabilities[item_index, int(predicted_indexes[item_index])])
            target = target_counts[item_index]
            rows.append(
                {
                    "image_id": str(image_id),
                    "target_count": target,
                    "predicted_count": prediction,
                    "confidence": confidence,
                    "absolute_error": abs(prediction - target),
                    "target_hypersegmentation": target >= hypersegmentation_threshold,
                    "predicted_hypersegmentation": prediction >= hypersegmentation_threshold,
                }
            )
            targets_all.append(target)
            predictions_all.append(prediction)

    return rows, targets_all, predictions_all


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Write metrics JSON and per-sample CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.weights_path, device=device)
    model, class_values, checkpoint_image_size = build_model_from_checkpoint(checkpoint, device)
    image_size = int(args.image_size or checkpoint_image_size)
    manifest_class_values = load_lobe_count_values(args.manifest)
    if class_values != manifest_class_values:
        print(f"Warning: checkpoint class values {class_values} differ from manifest {manifest_class_values}")

    dataset = NucleusLobeCountDataset(
        manifest_path=args.manifest,
        split=args.split,
        count_values=class_values,
        transform=get_lobe_count_val_transforms(image_size=image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows, targets, predictions = predict(
        model=model,
        dataloader=dataloader,
        class_values=class_values,
        device=device,
        hypersegmentation_threshold=args.hypersegmentation_threshold,
    )
    metrics = compute_lobe_count_metrics(
        targets=targets,
        predictions=predictions,
        count_values=class_values,
        hypersegmentation_threshold=args.hypersegmentation_threshold,
    )
    summary = asdict(metrics)
    summary["split"] = args.split
    summary["weights_path"] = to_project_relative_str(args.weights_path)
    write_outputs(args.output_dir, summary=summary, rows=rows)

    print(f"Evaluated split: {args.split}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Exact count accuracy: {summary['exact_accuracy']:.4f}")
    print(f"+/-1 accuracy: {summary['plus_minus_one_accuracy']:.4f}")
    print(
        "Binary hypersegmentation accuracy: "
        f"{summary['binary_hypersegmentation_accuracy']:.4f}"
    )
    print(f"Confusion matrix labels: {class_values}")
    print(f"Confusion matrix: {summary['confusion_matrix']}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")


if __name__ == "__main__":
    main()
