"""Обучает U-Net для бинарной сегментации ядра нейтрофила."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
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
from src.datasets.nucleus_segmentation_dataset import NucleusSegmentationDataset
from src.datasets.transforms import (
    get_segmentation_train_transforms,
    get_segmentation_val_transforms,
)
from src.models.unet import UNet
from src.utils.seed import set_seed

DEFAULT_MANIFEST_PATH = (
    PATHS.processed_data / "nucleus_segmentation" / "curated" / "manifest.csv"
)
DEFAULT_WEIGHTS_PATH = PATHS.models / "unet_nucleus.pt"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Train U-Net nucleus segmentation weights.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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


def dice_coefficient_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Вычисляет средний Dice по батчу из сырых логитов."""
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Дифференцируемая Dice-loss из сырых логитов."""
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Объединяет BCE и Dice-loss для малых масок переднего плана."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + soft_dice_loss(logits, targets)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Обучает одну эпоху."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    sample_count = 0

    for images, masks, _ in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        dice = dice_coefficient_from_logits(logits.detach(), masks)
        total_dice += float(dice.item()) * batch_size
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "dice": total_dice / sample_count,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Оценивает модель на валидационном разбиении."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    sample_count = 0

    for images, masks, _ in tqdm(dataloader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice_coefficient_from_logits(logits, masks).item()) * batch_size
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "dice": total_dice / sample_count,
    }


def _checkpoint_payload(
    model: UNet,
    epoch: int,
    best_val_dice: float,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Создает полезную нагрузку чекпоинта, которую можно загрузить для инференса."""
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "history": history,
        "model_config": {
            "in_channels": 3,
            "out_channels": 1,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def save_checkpoint(
    model: UNet,
    epoch: int,
    best_val_dice: float,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> None:
    """Сохраняет веса модели и сопутствующий JSON с метриками."""
    args.weights_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model=model,
        epoch=epoch,
        best_val_dice=best_val_dice,
        args=args,
        history=history,
    )
    torch.save(payload, args.weights_path)

    metrics_path = args.weights_path.with_suffix(".metrics.json")
    metrics_payload = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "history": history,
        "weights_path": to_project_relative_str(args.weights_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)

    train_dataset = NucleusSegmentationDataset(
        manifest_path=args.manifest,
        split="train",
        transform=get_segmentation_train_transforms(image_size=args.image_size),
    )
    val_dataset = NucleusSegmentationDataset(
        manifest_path=args.manifest,
        split="val",
        transform=get_segmentation_val_transforms(image_size=args.image_size),
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

    model = UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Device: {device}")

    best_val_dice = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                model=model,
                epoch=epoch,
                best_val_dice=best_val_dice,
                args=args,
                history=history,
            )
            print(f"Saved best weights: {to_project_relative_str(args.weights_path)}")


if __name__ == "__main__":
    main()
