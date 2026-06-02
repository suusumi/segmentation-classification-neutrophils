"""Оценивает U-Net для переднего плана и границ долей."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import load_lobe_count_values
from src.datasets.nucleus_lobe_segmentation_dataset import NucleusLobeSegmentationDataset
from src.datasets.transforms import get_lobe_segmentation_val_transforms
from src.models.unet import UNet
from src.pipeline.artifacts import save_mask
from src.pipeline.lobe_segmentation_postprocessing import postprocess_lobe_segmentation
from src.training.lobe_count_metrics import compute_lobe_count_metrics

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_WEIGHTS_PATH = PATHS.models / "lobe_boundary_unet.pt"
DEFAULT_OUTPUT_DIR = PATHS.outputs / "lobe_boundary_eval"
DEFAULT_PREDICTED_MASK_DIR = PATHS.processed_data / "nucleus_lobes" / "predicted_nucleus_masks"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Evaluate lobe foreground/boundary U-Net.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--foreground-threshold", type=float, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=0.2)
    parser.add_argument("--min-segment-area-px", type=int, default=None)
    parser.add_argument("--target-boundary-radius", type=int, default=None)
    parser.add_argument("--target-boundary-gap-radius", type=int, default=None)
    parser.add_argument("--hypersegmentation-threshold", type=int, default=None)
    parser.add_argument(
        "--nucleus-mask-source",
        choices=["cvat", "predicted", "mixed"],
        default="predicted",
        help="Nucleus mask input used during evaluation.",
    )
    parser.add_argument("--predicted-mask-dir", type=Path, default=DEFAULT_PREDICTED_MASK_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or another torch device string.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Преобразует имя устройства из CLI в устройство torch."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_checkpoint(weights_path: Path, device: torch.device) -> dict[str, Any]:
    """Загружает чекпоинт границ долей."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported lobe boundary checkpoint format: {weights_path}")
    return cast(dict[str, Any], checkpoint)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[UNet, int]:
    """Создает U-Net из чекпоинта."""
    model_config = checkpoint.get("model_config", {})
    model = UNet(
        in_channels=int(model_config.get("in_channels", 4)),
        out_channels=int(model_config.get("out_channels", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, int(model_config.get("image_size", 256))


def _setting(
    cli_value: float | int | None,
    checkpoint: dict[str, Any],
    name: str,
    default: float | int,
) -> float | int:
    """Разрешает переопределение из CLI или настройку постобработки из чекпоинта."""
    if cli_value is not None:
        return cli_value
    return cast(float | int, checkpoint.get("postprocessing", {}).get(name, default))


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Вычисляет бинарную метрику Dice."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


def save_labeled_mask(mask: np.ndarray, path: Path) -> Path:
    """Сохраняет небольшую детерминированную цветовую визуализацию размеченных компонентов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(
        [
            [0, 0, 0],
            [220, 30, 30],
            [30, 160, 220],
            [40, 190, 90],
            [230, 180, 40],
            [160, 90, 230],
            [240, 100, 160],
        ],
        dtype=np.uint8,
    )
    colored = palette[np.mod(mask.astype(np.int32), len(palette))]
    Image.fromarray(colored).save(path)
    return path


@torch.no_grad()
def evaluate(
    model: UNet,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    settings: dict[str, float | int],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Запускает оценку и сохраняет маски для каждого образца."""
    rows: list[dict[str, Any]] = []
    targets_all: list[int] = []
    predictions_all: list[int] = []
    foreground_dir = output_dir / "foreground"
    boundary_dir = output_dir / "boundary"
    split_dir = output_dir / "split_components"

    model.eval()
    for inputs, targets, segment_counts, image_ids in tqdm(dataloader, desc="evaluate"):
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        target_array = targets.detach().cpu().numpy()

        for item_index, image_id in enumerate(image_ids):
            foreground_probability = probabilities[item_index, 0]
            boundary_probability = probabilities[item_index, 1]
            result = postprocess_lobe_segmentation(
                foreground_probability=foreground_probability,
                boundary_probability=boundary_probability,
                foreground_threshold=float(settings["foreground_threshold"]),
                boundary_threshold=float(settings["boundary_threshold"]),
                min_segment_area_px=int(settings["min_segment_area_px"]),
                support_mask=inputs[item_index, 3].detach().cpu().numpy() > 0.5,
            )
            target_count = int(segment_counts[item_index].item())
            predicted_count = result.segment_count
            image_key = str(image_id)
            foreground_path = save_mask(
                result.foreground_mask,
                foreground_dir / f"{image_key}.png",
            )
            boundary_path = save_mask(
                result.boundary_mask,
                boundary_dir / f"{image_key}.png",
            )
            split_path = save_labeled_mask(
                result.split_mask,
                split_dir / f"{image_key}.png",
            )
            rows.append(
                {
                    "image_id": image_key,
                    "target_count": target_count,
                    "predicted_count": predicted_count,
                    "absolute_error": abs(predicted_count - target_count),
                    "foreground_dice": dice_score(
                        result.foreground_mask,
                        target_array[item_index, 0],
                    ),
                    "boundary_dice": dice_score(
                        result.boundary_mask,
                        target_array[item_index, 1],
                    ),
                    "target_hypersegmentation": (
                        target_count >= int(settings["hypersegmentation_threshold"])
                    ),
                    "predicted_hypersegmentation": (
                        predicted_count >= int(settings["hypersegmentation_threshold"])
                    ),
                    "foreground_path": to_project_relative_str(foreground_path),
                    "boundary_path": to_project_relative_str(boundary_path),
                    "split_components_path": to_project_relative_str(split_path),
                }
            )
            targets_all.append(target_count)
            predictions_all.append(predicted_count)

    return rows, targets_all, predictions_all


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Записывает JSON с метриками и CSV."""
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
    """Точка входа CLI."""

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.weights_path, device=device)
    model, checkpoint_image_size = build_model_from_checkpoint(checkpoint, device=device)
    image_size = int(args.image_size or checkpoint_image_size)
    settings = {
        "foreground_threshold": _setting(
            args.foreground_threshold,
            checkpoint,
            "foreground_threshold",
            0.5,
        ),
        "boundary_threshold": _setting(args.boundary_threshold, checkpoint, "boundary_threshold", 0.5),
        "min_segment_area_px": _setting(
            args.min_segment_area_px,
            checkpoint,
            "min_segment_area_px",
            32,
        ),
        "target_boundary_radius": _setting(
            args.target_boundary_radius,
            checkpoint,
            "target_boundary_radius",
            1,
        ),
        "target_boundary_gap_radius": _setting(
            args.target_boundary_gap_radius,
            checkpoint,
            "target_boundary_gap_radius",
            6,
        ),
        "hypersegmentation_threshold": _setting(
            args.hypersegmentation_threshold,
            checkpoint,
            "hypersegmentation_threshold",
            5,
        ),
    }
    dataset = NucleusLobeSegmentationDataset(
        manifest_path=args.manifest,
        split=args.split,
        transform=get_lobe_segmentation_val_transforms(image_size=image_size),
        boundary_radius=int(settings["target_boundary_radius"]),
        boundary_gap_radius=int(settings["target_boundary_gap_radius"]),
        nucleus_mask_source=args.nucleus_mask_source,
        predicted_mask_dir=args.predicted_mask_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows, targets, predictions = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
        settings=settings,
    )
    count_values = sorted(set(load_lobe_count_values(args.manifest)).union(predictions))
    metrics = compute_lobe_count_metrics(
        targets=targets,
        predictions=predictions,
        count_values=count_values,
        hypersegmentation_threshold=int(settings["hypersegmentation_threshold"]),
    )
    summary = asdict(metrics)
    summary["split"] = args.split
    summary["weights_path"] = to_project_relative_str(args.weights_path)
    summary["settings"] = settings
    summary["nucleus_mask_source"] = args.nucleus_mask_source
    summary["predicted_mask_dir"] = to_project_relative_str(args.predicted_mask_dir)
    summary["foreground_dice_mean"] = float(np.mean([row["foreground_dice"] for row in rows]))
    summary["boundary_dice_mean"] = float(np.mean([row["boundary_dice"] for row in rows]))
    write_outputs(args.output_dir, summary=summary, rows=rows)

    print(f"Evaluated split: {args.split}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Exact count accuracy: {summary['exact_accuracy']:.4f}")
    print(f"+/-1 accuracy: {summary['plus_minus_one_accuracy']:.4f}")
    print(
        "Binary hypersegmentation accuracy: "
        f"{summary['binary_hypersegmentation_accuracy']:.4f}"
    )
    print(f"Foreground Dice mean: {summary['foreground_dice_mean']:.4f}")
    print(f"Boundary Dice mean: {summary['boundary_dice_mean']:.4f}")
    print(f"Confusion matrix labels: {summary['count_values']}")
    print(f"Confusion matrix: {summary['confusion_matrix']}")
    print(f"Output directory: {to_project_relative_str(args.output_dir)}")


if __name__ == "__main__":
    main()
