"""EEGNet — Lawhern et al. 2018 [36]."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., J. Neural Eng. 2018).

    Default: F1=8, D=2, F2=16, kernel_length=64.
    Input: (batch, time, channels).
    """

    def __init__(
        self,
        n_channels: int = 15,
        n_samples: int = 251,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2
        # Block 1
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        # Block 2
        self.sep_conv = nn.Conv2d(
            F1 * D, F2, (1, 16), padding=(0, 8), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            flat = self._forward_features(
                torch.zeros(1, n_samples, n_channels)
            ).shape[1]
        self.fc = nn.Linear(flat, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._forward_features(x))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EEGNet":
        return cls(
            n_channels=cfg.get("input_channels", 15),
            n_samples=cfg.get("input_time", 251),
            F1=cfg.get("F1", 8),
            D=cfg.get("D", 2),
            F2=cfg.get("F2", 16),
            kernel_length=cfg.get("kernel_length", 64),
            dropout=cfg.get("dropout", 0.5),
            num_classes=cfg.get("num_classes", 2),
        )
