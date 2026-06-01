"""CNN-SE-BiLSTM — Sun et al. 2026 Table 1 replication.

Table 1 pipeline:
  CNN (Conv1 16×(10,10) → MaxPool (10,1) → Conv2 12×(15,15) → SE)
  → Flatten 13080
  → Bi-LSTM (16 units, bidirectional → 32, tanh)
  → FC + tanh → FC 2 (logits; softmax at inference)

§2.5.5 describes Conv1D then Conv2D; kernels are 2D (10×10) and (15×15) in Table 1,
implemented as Conv2d on (time × channel) plane, consistent with the figure.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.models.paper_cnn_frontend import PAPER_FLATTEN_SIZE, PaperCNNSEFrontend

logger = logging.getLogger(__name__)


class CNNSEBiLSTM(nn.Module):
    """Hybrid CNN-SE-BiLSTM per Sun et al. 2026."""

    def __init__(
        self,
        input_time: int = 251,
        input_channels: int = 15,
        conv1_filters: int = 16,
        conv1_kernel: tuple[int, int] = (10, 10),
        conv2_filters: int = 12,
        conv2_kernel: tuple[int, int] = (15, 15),
        pool_kernel: tuple[int, int] = (10, 1),
        se_reduction: int = 3,
        lstm_hidden: int = 16,
        bidirectional: bool = True,
        num_classes: int = 2,
        flatten_size: int = PAPER_FLATTEN_SIZE,
        log_shapes: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.log_shapes = log_shapes
        self.cnn_se = PaperCNNSEFrontend(
            input_time=input_time,
            input_channels=input_channels,
            conv1_filters=conv1_filters,
            conv1_kernel=conv1_kernel,
            conv2_filters=conv2_filters,
            conv2_kernel=conv2_kernel,
            pool_kernel=pool_kernel,
            se_reduction=se_reduction,
            target_flatten=flatten_size,
            strict_flatten=True,
        )
        # Table 1: single Bi-LSTM step on flattened vector (seq_len=1, input=13080)
        self.lstm = nn.LSTM(
            input_size=self.cnn_se.flatten_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.fc_tanh = nn.Linear(lstm_out, lstm_out)
        self.fc_out = nn.Linear(lstm_out, num_classes)

    def _log(self, name: str, t: torch.Tensor) -> None:
        if self.log_shapes:
            logger.info("%s: %s", name, tuple(t.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, num_classes). No softmax in forward (CrossEntropyLoss)."""
        flat = self.cnn_se(x)  # (B, 13080)
        self._log("flatten", flat)
        seq = flat.unsqueeze(1)  # (B, 1, 13080)
        out, _ = self.lstm(seq)
        feat = out[:, 0, :]
        self._log("bilstm", feat)
        feat = torch.tanh(self.fc_tanh(feat))
        logits = self.fc_out(feat)
        self._log("logits", logits)
        return logits

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CNNSEBiLSTM":
        ck1 = cfg.get("conv1_kernel", [10, 10])
        ck2 = cfg.get("conv2_kernel", [15, 15])
        pk = cfg.get("pool_kernel", [10, 1])
        return cls(
            input_time=cfg.get("input_time", 251),
            input_channels=cfg.get("input_channels", 15),
            conv1_filters=cfg.get("conv1_filters", 16),
            conv1_kernel=(int(ck1[0]), int(ck1[1])),
            conv2_filters=cfg.get("conv2_filters", 12),
            conv2_kernel=(int(ck2[0]), int(ck2[1])),
            pool_kernel=(int(pk[0]), int(pk[1])),
            se_reduction=cfg.get("se_reduction", 3),
            lstm_hidden=cfg.get("lstm_hidden", 16),
            bidirectional=cfg.get("bidirectional", True),
            num_classes=cfg.get("num_classes", 2),
            flatten_size=cfg.get("flatten_size", PAPER_FLATTEN_SIZE),
            log_shapes=cfg.get("log_shapes", False),
        )
