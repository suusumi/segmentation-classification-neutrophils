"""Train a simple nucleus lobe count baseline."""

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
from src.datasets.nucleus_lobe_count_dataset import (
    NucleusLobeCountDataset,
    load_lobe_count_values,
)
from src.datasets.transforms import (
    get_lobe_count_train_transforms,
    get_lobe_count_val_transforms,
)
from src.models import LobeCountCNN
from src.training.lobe_count_metrics import compute_lobe_count_metrics
from src.utils.seed import set_seed

DEFAULT_MANIFEST_PATH = PATHS.processed_data / "nucleus_lobes" / "curated" / "manifest.csv"
DEFAULT_WEIGHTS_PATH = PATHS.models / "lobe_count_baseline.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a nucleus lobe count classifier.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def build_class_weights(
    dataset: NucleusLobeCountDataset,
    device: torch.device,
) -> torch.Tensor:
    """Build inverse-frequency class weights from the training split."""

    counts_by_index = torch.zeros(len(dataset.count_values), dtype=torch.float32)
    for sample in dataset.samples:
        counts_by_index[dataset.count_to_index[sample.segment_count]] += 1

    total = counts_by_index.sum()
    weights = torch.zeros_like(counts_by_index)
    nonzero = counts_by_index > 0
    weights[nonzero] = total / (float(len(dataset.count_values)) * counts_by_index[nonzero])
    return weights.to(device)


def _counts_from_logits(logits: torch.Tensor, count_values: list[int]) -> list[int]:
    """Convert model logits into raw segment counts."""

    predicted_indexes = logits.argmax(dim=1).detach().cpu().tolist()
    return [count_values[int(index)] for index in predicted_indexes]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""

    model.train()
    total_loss = 0.0
    sample_count = 0

    for inputs, targets, _, _ in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
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
    criterion: nn.Module,
    count_values: list[int],
    device: torch.device,
    hypersegmentation_threshold: int,
) -> dict[str, Any]:
    """Evaluate the model and return loss plus count metrics."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for inputs, targets, segment_counts, _ in tqdm(dataloader, desc="val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        targets_all.extend(int(value) for value in segment_counts.cpu().tolist())
        predictions_all.extend(_counts_from_logits(logits, count_values=count_values))

    metrics = compute_lobe_count_metrics(
        targets=targets_all,
        predictions=predictions_all,
        count_values=count_values,
        hypersegmentation_threshold=hypersegmentation_threshold,
    )
    result = asdict(metrics)
    result["loss"] = total_loss / sample_count
    return result


def _checkpoint_payload(
    model: LobeCountCNN,
    epoch: int,
    best_val_exact_accuracy: float,
    args: argparse.Namespace,
    class_values: list[int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a checkpoint payload for lobe count inference."""

    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_exact_accuracy": best_val_exact_accuracy,
        "history": history,
        "class_values": class_values,
        "model_config": {
            "in_channels": 4,
            "num_classes": len(class_values),
            "base_channels": args.base_channels,
            "dropout": args.dropout,
            "image_size": args.image_size,
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def save_checkpoint(
    model: LobeCountCNN,
    epoch: int,
    best_val_exact_accuracy: float,
    args: argparse.Namespace,
    class_values: list[int],
    history: list[dict[str, Any]],
) -> None:
    """Save model weights and a JSON sidecar."""

    args.weights_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model=model,
        epoch=epoch,
        best_val_exact_accuracy=best_val_exact_accuracy,
        args=args,
        class_values=class_values,
        history=history,
    )
    torch.save(payload, args.weights_path)

    metrics_path = args.weights_path.with_suffix(".metrics.json")
    metrics_payload = {
        "epoch": epoch,
        "best_val_exact_accuracy": best_val_exact_accuracy,
        "history": history,
        "weights_path": to_project_relative_str(args.weights_path),
        "class_values": class_values,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    class_values = load_lobe_count_values(args.manifest)

    train_dataset = NucleusLobeCountDataset(
        manifest_path=args.manifest,
        split="train",
        count_values=class_values,
        transform=get_lobe_count_train_transforms(image_size=args.image_size),
    )
    val_dataset = NucleusLobeCountDataset(
        manifest_path=args.manifest,
        split="val",
        count_values=class_values,
        transform=get_lobe_count_val_transforms(image_size=args.image_size),
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

    model = LobeCountCNN(
        in_channels=4,
        num_classes=len(class_values),
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=build_class_weights(train_dataset, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class values: {class_values}")
    print(f"Device: {device}")

    best_val_exact_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            count_values=class_values,
            device=device,
            hypersegmentation_threshold=args.hypersegmentation_threshold,
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
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"exact={row['val_exact_accuracy']:.4f} "
            f"pm1={row['val_plus_minus_one_accuracy']:.4f} "
            f"hyper={row['val_binary_hypersegmentation_accuracy']:.4f}"
        )

        if val_metrics["exact_accuracy"] > best_val_exact_accuracy:
            best_val_exact_accuracy = float(val_metrics["exact_accuracy"])
            save_checkpoint(
                model=model,
                epoch=epoch,
                best_val_exact_accuracy=best_val_exact_accuracy,
                args=args,
                class_values=class_values,
                history=history,
            )
            print(f"Saved best weights: {to_project_relative_str(args.weights_path)}")


if __name__ == "__main__":
    main()
