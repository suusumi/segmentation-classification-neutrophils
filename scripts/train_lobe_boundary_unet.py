"""Train U-Net to predict nucleus lobe foreground and separating boundaries."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import PATHS, to_project_relative_str
from src.datasets.nucleus_lobe_count_dataset import load_lobe_count_values
from src.datasets.nucleus_lobe_segmentation_dataset import NucleusLobeSegmentationDataset
from src.datasets.transforms import (
    get_lobe_segmentation_train_transforms,
    get_lobe_segmentation_val_transforms,
)
from src.models import UNet
from src.training.lobe_count_metrics import compute_lobe_count_metrics
from src.training.lobe_segmentation_postprocessing import postprocess_lobe_segmentation
from src.utils.seed import set_seed

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_WEIGHTS_PATH = PATHS.models / "lobe_boundary_unet.pt"
DEFAULT_PREDICTED_MASK_DIR = PATHS.processed_data / "nucleus_lobes" / "predicted_nucleus_masks"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train lobe foreground/boundary U-Net.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--boundary-loss-weight", type=float, default=2.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-threshold", type=float, default=0.2)
    parser.add_argument("--min-segment-area-px", type=int, default=32)
    parser.add_argument("--target-boundary-radius", type=int, default=1)
    parser.add_argument("--target-boundary-gap-radius", type=int, default=6)
    parser.add_argument(
        "--train-nucleus-mask-source",
        choices=["cvat", "predicted", "mixed"],
        default="mixed",
        help="Nucleus mask input used for training.",
    )
    parser.add_argument(
        "--val-nucleus-mask-source",
        choices=["cvat", "predicted", "mixed"],
        default="predicted",
        help="Nucleus mask input used for validation.",
    )
    parser.add_argument("--predicted-mask-dir", type=Path, default=DEFAULT_PREDICTED_MASK_DIR)
    parser.add_argument("--predicted-mask-probability", type=float, default=0.7)
    parser.add_argument("--mask-noise-probability", type=float, default=0.5)
    parser.add_argument("--mask-noise-max-radius", type=int, default=2)
    parser.add_argument(
        "--init-weights-path",
        type=Path,
        default=None,
        help="Optional lobe boundary checkpoint to fine-tune from.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute differentiable Dice loss per channel."""

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def lobe_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    boundary_loss_weight: float,
) -> torch.Tensor:
    """Combine BCE and Dice losses for foreground and boundary channels."""

    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_by_channel = bce.mean(dim=(0, 2, 3))
    dice_by_channel = soft_dice_loss(logits, targets)
    channel_loss = bce_by_channel + dice_by_channel
    return channel_loss[0] + boundary_loss_weight * channel_loss[1]


def _counts_from_logits(
    logits: torch.Tensor,
    support_masks: torch.Tensor,
    foreground_threshold: float,
    boundary_threshold: float,
    min_segment_area_px: int,
) -> list[int]:
    """Postprocess batch logits into predicted lobe counts."""

    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    support_array = support_masks.detach().cpu().numpy() > 0.5
    predictions: list[int] = []
    for item, support_mask in zip(probabilities, support_array, strict=True):
        result = postprocess_lobe_segmentation(
            foreground_probability=item[0],
            boundary_probability=item[1],
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            min_segment_area_px=min_segment_area_px,
            support_mask=support_mask,
        )
        predictions.append(result.segment_count)
    return predictions


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    boundary_loss_weight: float,
) -> float:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0
    sample_count = 0
    for inputs, targets, _, _ in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = lobe_segmentation_loss(
            logits=logits,
            targets=targets,
            boundary_loss_weight=boundary_loss_weight,
        )
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    return total_loss / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    count_values: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate validation loss and count metrics."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for inputs, targets, segment_counts, _ in tqdm(dataloader, desc="val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = lobe_segmentation_loss(
            logits=logits,
            targets=targets,
            boundary_loss_weight=args.boundary_loss_weight,
        )

        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        targets_all.extend(int(value) for value in segment_counts.cpu().tolist())
        predictions_all.extend(
            _counts_from_logits(
                logits=logits,
                support_masks=inputs[:, 3],
                foreground_threshold=args.foreground_threshold,
                boundary_threshold=args.boundary_threshold,
                min_segment_area_px=args.min_segment_area_px,
            )
        )

    metric_labels = sorted(set(count_values).union(predictions_all))
    metrics = compute_lobe_count_metrics(
        targets=targets_all,
        predictions=predictions_all,
        count_values=metric_labels,
        hypersegmentation_threshold=args.hypersegmentation_threshold,
    )
    result = asdict(metrics)
    result["loss"] = total_loss / sample_count
    result["predicted_count_distribution"] = {
        str(count): predictions_all.count(count)
        for count in sorted(set(predictions_all))
    }
    return result


def _checkpoint_payload(
    model: UNet,
    epoch: int,
    best_val_exact_accuracy: float,
    args: argparse.Namespace,
    count_values: list[int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a checkpoint payload."""

    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_exact_accuracy": best_val_exact_accuracy,
        "history": history,
        "count_values": count_values,
        "model_config": {
            "in_channels": 4,
            "out_channels": 2,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
        },
        "postprocessing": {
            "foreground_threshold": args.foreground_threshold,
            "boundary_threshold": args.boundary_threshold,
            "min_segment_area_px": args.min_segment_area_px,
            "target_boundary_radius": args.target_boundary_radius,
            "target_boundary_gap_radius": args.target_boundary_gap_radius,
            "hypersegmentation_threshold": args.hypersegmentation_threshold,
        },
        "nucleus_mask_training": {
            "train_nucleus_mask_source": args.train_nucleus_mask_source,
            "val_nucleus_mask_source": args.val_nucleus_mask_source,
            "predicted_mask_dir": to_project_relative_str(args.predicted_mask_dir),
            "predicted_mask_probability": args.predicted_mask_probability,
            "mask_noise_probability": args.mask_noise_probability,
            "mask_noise_max_radius": args.mask_noise_max_radius,
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def save_checkpoint(
    model: UNet,
    epoch: int,
    best_val_exact_accuracy: float,
    args: argparse.Namespace,
    count_values: list[int],
    history: list[dict[str, Any]],
) -> None:
    """Save model weights and sidecar metrics."""

    args.weights_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model=model,
        epoch=epoch,
        best_val_exact_accuracy=best_val_exact_accuracy,
        args=args,
        count_values=count_values,
        history=history,
    )
    torch.save(payload, args.weights_path)
    metrics_path = args.weights_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "best_val_exact_accuracy": best_val_exact_accuracy,
                "history": history,
                "weights_path": to_project_relative_str(args.weights_path),
                "count_values": count_values,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    count_values = load_lobe_count_values(args.manifest)

    train_dataset = NucleusLobeSegmentationDataset(
        manifest_path=args.manifest,
        split="train",
        transform=get_lobe_segmentation_train_transforms(image_size=args.image_size),
        boundary_radius=args.target_boundary_radius,
        boundary_gap_radius=args.target_boundary_gap_radius,
        nucleus_mask_source=args.train_nucleus_mask_source,
        predicted_mask_dir=args.predicted_mask_dir,
        predicted_mask_probability=args.predicted_mask_probability,
        mask_noise_probability=args.mask_noise_probability,
        mask_noise_max_radius=args.mask_noise_max_radius,
    )
    val_dataset = NucleusLobeSegmentationDataset(
        manifest_path=args.manifest,
        split="val",
        transform=get_lobe_segmentation_val_transforms(image_size=args.image_size),
        boundary_radius=args.target_boundary_radius,
        boundary_gap_radius=args.target_boundary_gap_radius,
        nucleus_mask_source=args.val_nucleus_mask_source,
        predicted_mask_dir=args.predicted_mask_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = UNet(in_channels=4, out_channels=2, base_channels=args.base_channels).to(device)
    if args.init_weights_path is not None:
        checkpoint = torch.load(args.init_weights_path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Unsupported init checkpoint: {args.init_weights_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized from: {to_project_relative_str(args.init_weights_path)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train nucleus mask source: {args.train_nucleus_mask_source}")
    print(f"Val nucleus mask source: {args.val_nucleus_mask_source}")
    print(f"Predicted mask dir: {to_project_relative_str(args.predicted_mask_dir)}")
    print(f"Count values: {count_values}")
    print(f"Device: {device}")

    best_val_exact_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            boundary_loss_weight=args.boundary_loss_weight,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            count_values=count_values,
            args=args,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_exact_accuracy": val_metrics["exact_accuracy"],
            "val_plus_minus_one_accuracy": val_metrics["plus_minus_one_accuracy"],
            "val_binary_hypersegmentation_accuracy": (
                val_metrics["binary_hypersegmentation_accuracy"]
            ),
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "val_count_values": val_metrics["count_values"],
            "val_predicted_count_distribution": (
                val_metrics["predicted_count_distribution"]
            ),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} val_loss={row['val_loss']:.4f} "
            f"exact={row['val_exact_accuracy']:.4f} "
            f"pm1={row['val_plus_minus_one_accuracy']:.4f} "
            f"hyper={row['val_binary_hypersegmentation_accuracy']:.4f} "
            f"pred={row['val_predicted_count_distribution']}"
        )

        if val_metrics["exact_accuracy"] > best_val_exact_accuracy:
            best_val_exact_accuracy = float(val_metrics["exact_accuracy"])
            save_checkpoint(
                model=model,
                epoch=epoch,
                best_val_exact_accuracy=best_val_exact_accuracy,
                args=args,
                count_values=count_values,
                history=history,
            )
            print(f"Saved best weights: {to_project_relative_str(args.weights_path)}")


if __name__ == "__main__":
    main()
