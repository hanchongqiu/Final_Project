"""Baseline 20-Day CNN architecture matching Jiang, Kelly, & Xiu (2023) Figure 2."""

from __future__ import annotations

import torch
from torch import nn


class CNN20D(nn.Module):
    """Exact 20-day chart CNN (`FC 46080`) specified in Final Project.pdf Figure 2.

    Input shape : `(B, 1, 64, 60)` float32 in `[0.0, 1.0]`
    Block 1     : Conv2d(1->64, k=(5,3), s=(3,1), d=(2,1), p=(12,1)) -> BN -> LReLU -> MaxPool(2,1) => (64, 13, 60)
    Block 2     : Conv2d(64->128, k=(5,3), s=(1,1), p=(2,1))         -> BN -> LReLU -> MaxPool(2,1) => (128, 6, 60)
    Block 3     : Conv2d(128->256, k=(5,3), s=(1,1), p=(2,1))        -> BN -> LReLU -> MaxPool(2,1) => (256, 3, 60)
    Head        : Flatten(46080) -> Dropout(0.5) -> Linear(46080, 2)
    """

    FC_IN_FEATURES = 256 * 3 * 60  # 46,080

    def __init__(self, dropout: float = 0.5, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=(5, 3),
                stride=(3, 1),
                dilation=(2, 1),
                padding=(12, 1),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.FC_IN_FEATURES, 2),
        )
        self.reset_parameters()

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        """Target convolutional layer for Grad-CAM attribution (Member D)."""
        layer = self.block3[0]
        assert isinstance(layer, nn.Conv2d)
        return layer

    def reset_parameters(self) -> None:
        """Apply Xavier uniform initialization to Conv2d and Linear layers per JKX (2023)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
