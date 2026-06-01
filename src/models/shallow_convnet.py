"""ShallowConvNet — Schirrmeister et al. 2017 [37] for EEG decoding."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet from Schirrmeister et al. (Hum. Brain Mapp. 2017).

    Input: (batch, time, channels) -> internally (batch, 1, channels, time).
    """

    def __init__(
        self,
        n_channels: int = 15,
        n_times: int = 251,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_width: int = 75,
        pool_time_stride: int = 15,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (1, filter_time_length), bias=False
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, (n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_spat, affine=True)
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_width), stride=(1, pool_time_stride)
        )
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            x = torch.zeros(1, n_times, n_channels)
            flat = self._features(x).shape[1]
        self.fc = nn.Linear(flat, num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, T)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = torch.square(x)
        x = self.bn(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._features(x))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ShallowConvNet":
        return cls(
            n_channels=cfg.get("input_channels", 15),
            n_times=cfg.get("input_time", 251),
            num_classes=cfg.get("num_classes", 2),
            dropout=cfg.get("dropout", 0.5),
        )
