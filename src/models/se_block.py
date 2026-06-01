"""Squeeze-and-Excitation block."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Channel attention via global average pooling and gating."""

    def __init__(self, channels: int, reduction: int = 3) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y
