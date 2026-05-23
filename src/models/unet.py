"""Minimal U-Net architecture for future nucleus segmentation training."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(tensor))


class UNet(nn.Module):
    """Small U-Net suitable for binary nucleus segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(tensor)
        down2 = self.down2(self.pool1(down1))
        bridge = self.bridge(self.pool2(down2))
        up2 = self.up2(bridge)
        up2 = torch.cat([up2, down2], dim=1)
        up2 = self.conv2(up2)
        up1 = self.up1(up2)
        up1 = torch.cat([up1, down1], dim=1)
        up1 = self.conv1(up1)
        return cast(torch.Tensor, self.output(up1))
