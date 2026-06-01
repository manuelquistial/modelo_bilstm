"""CNN-LSTM baseline — Table 1 CNN without SE (Sun et al. §2.6, ref [40])."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.paper_cnn_frontend import (
    PAPER_FLATTEN_SIZE,
    _build_frontend,
    discover_paddings_for_paper_flatten,
)


class CNNLSTM(nn.Module):
    """CNN + Bi-LSTM without SE; same flatten + LSTM head as proposed model."""

    def __init__(
        self,
        input_time: int = 251,
        input_channels: int = 15,
        conv1_filters: int = 16,
        conv1_kernel: tuple[int, int] = (10, 10),
        conv2_filters: int = 12,
        conv2_kernel: tuple[int, int] = (15, 15),
        pool_kernel: tuple[int, int] = (10, 1),
        lstm_hidden: int = 16,
        bidirectional: bool = True,
        num_classes: int = 2,
        flatten_size: int = PAPER_FLATTEN_SIZE,
    ) -> None:
        super().__init__()
        pad1, pad2 = discover_paddings_for_paper_flatten(
            input_time,
            input_channels,
            conv1_filters,
            conv1_kernel,
            conv2_filters,
            conv2_kernel,
            pool_kernel,
            3,
            flatten_size,
            use_se=False,
        )
        self.cnn = _build_frontend(
            conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
            pool_kernel, 3, pad1, pad2, use_se=False,
        )
        with torch.no_grad():
            d = int(self.cnn(torch.zeros(1, 1, input_time, input_channels)).numel())
        self.proj = nn.Linear(d, flatten_size) if d != flatten_size else nn.Identity()
        self.lstm = nn.LSTM(
            flatten_size, lstm_hidden, batch_first=True, bidirectional=bidirectional
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc_tanh = nn.Linear(out_dim, out_dim)
        self.fc_out = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.flatten(self.cnn(x), 1)
        x = self.proj(x)
        out, _ = self.lstm(x.unsqueeze(1))
        feat = torch.tanh(self.fc_tanh(out[:, 0, :]))
        return self.fc_out(feat)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CNNLSTM":
        ck1 = cfg.get("conv1_kernel", [10, 10])
        ck2 = cfg.get("conv2_kernel", [15, 15])
        pk = cfg.get("pool_kernel", [10, 1])
        return cls(
            input_time=cfg.get("input_time", 251),
            input_channels=cfg.get("input_channels", 15),
            conv1_kernel=(int(ck1[0]), int(ck1[1])),
            conv2_kernel=(int(ck2[0]), int(ck2[1])),
            pool_kernel=(int(pk[0]), int(pk[1])),
            conv1_filters=cfg.get("conv1_filters", 16),
            conv2_filters=cfg.get("conv2_filters", 12),
            lstm_hidden=cfg.get("lstm_hidden", 16),
            bidirectional=cfg.get("bidirectional", True),
            num_classes=cfg.get("num_classes", 2),
            flatten_size=cfg.get("flatten_size", PAPER_FLATTEN_SIZE),
        )
