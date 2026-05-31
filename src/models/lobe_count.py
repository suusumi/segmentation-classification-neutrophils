"""Small CNN baseline for nucleus lobe counting."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional downsampling block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(tensor))


class LobeCountCNN(nn.Module):
    """Compact classifier for segment count from RGB image plus nucleus mask."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 24,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(tensor)
        return cast(torch.Tensor, self.classifier(features))
