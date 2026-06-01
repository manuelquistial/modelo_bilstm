"""CNN + SE frontend matching Sun et al. 2026 Table 1.

Architecture (paper §2.5.5, Table 1):
  Input (251, 15)  -> tensor (B, 1, T, C)
  Conv_1: 16 filters, kernel (10, 10), ReLU + BatchNorm
  MaxPool: kernel (10, 1)
  Conv_2: 12 filters, kernel (15, 15), ReLU + BatchNorm
  SE: GAP + FC(C/r) + FC(C) + sigmoid, r=3
  Flatten -> 13,080 features (Table 1)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.models.se_block import SEBlock

logger = logging.getLogger(__name__)

PAPER_FLATTEN_SIZE = 13080


def _build_frontend(
    conv1_filters: int,
    conv1_kernel: tuple[int, int],
    conv2_filters: int,
    conv2_kernel: tuple[int, int],
    pool_kernel: tuple[int, int],
    se_reduction: int,
    conv1_padding: tuple[int, int],
    conv2_padding: tuple[int, int],
    use_se: bool = True,
) -> nn.Module:
    layers: list[nn.Module] = [
        nn.Conv2d(1, conv1_filters, conv1_kernel, padding=conv1_padding),
        nn.BatchNorm2d(conv1_filters),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel),
        nn.Conv2d(conv1_filters, conv2_filters, conv2_kernel, padding=conv2_padding),
        nn.BatchNorm2d(conv2_filters),
        nn.ReLU(inplace=True),
    ]
    if use_se:
        layers.append(SEBlock(conv2_filters, reduction=se_reduction))
    return nn.Sequential(*layers)


def _flatten_dim(
    input_time: int,
    input_channels: int,
    conv1_filters: int,
    conv1_kernel: tuple[int, int],
    conv2_filters: int,
    conv2_kernel: tuple[int, int],
    pool_kernel: tuple[int, int],
    se_reduction: int,
    conv1_padding: tuple[int, int],
    conv2_padding: tuple[int, int],
) -> int:
    front = _build_frontend(
        conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
        pool_kernel, se_reduction, conv1_padding, conv2_padding, use_se=True,
    )
    with torch.no_grad():
        x = torch.zeros(1, 1, input_time, input_channels)
        y = front(x)
    return int(y.numel())


def discover_paddings_for_paper_flatten(
    input_time: int = 251,
    input_channels: int = 15,
    conv1_filters: int = 16,
    conv1_kernel: tuple[int, int] = (10, 10),
    conv2_filters: int = 12,
    conv2_kernel: tuple[int, int] = (15, 15),
    pool_kernel: tuple[int, int] = (10, 1),
    se_reduction: int = 3,
    target: int = PAPER_FLATTEN_SIZE,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Search symmetric paddings so flatten size matches Table 1 (13080).

    MATLAB 'same' padding in the original study may differ slightly; this search
    finds the closest valid PyTorch configuration for replication.
    """
    best: tuple[tuple[int, int], tuple[int, int], int] | None = None
    for p1 in range(0, 20):
        for p2 in range(0, 20):
            pad1 = (p1, p1)
            pad2 = (p2, p2)
            try:
                dim = _flatten_dim(
                    input_time, input_channels,
                    conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
                    pool_kernel, se_reduction, pad1, pad2,
                )
            except Exception:
                continue
            if dim == target:
                logger.info("Paper flatten %d matched with conv1_pad=%s conv2_pad=%s", target, pad1, pad2)
                return pad1, pad2
            if best is None or abs(dim - target) < abs(best[2] - target):
                best = (pad1, pad2, dim)
    if best is not None and best[2] != target:
        logger.warning(
            "Could not reach flatten=%d exactly; using pad1=%s pad2=%s (flatten=%d). "
            "Verify against authors' MATLAB if metrics diverge.",
            target, best[0], best[1], best[2],
        )
        return best[0], best[1]
    return (4, 4), (7, 7)


class PaperCNNSEFrontend(nn.Module):
    """CNN + SE producing a fixed-size flattened vector for Bi-LSTM."""

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
        conv1_padding: tuple[int, int] | None = None,
        conv2_padding: tuple[int, int] | None = None,
        target_flatten: int = PAPER_FLATTEN_SIZE,
        strict_flatten: bool = True,
    ) -> None:
        super().__init__()
        self.input_time = input_time
        self.input_channels = input_channels
        self.target_flatten = target_flatten

        if conv1_padding is None or conv2_padding is None:
            conv1_padding, conv2_padding = discover_paddings_for_paper_flatten(
                input_time, input_channels,
                conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
                pool_kernel, se_reduction, target_flatten,
            )
        self.conv1_padding = conv1_padding
        self.conv2_padding = conv2_padding

        self.frontend = _build_frontend(
            conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
            pool_kernel, se_reduction, conv1_padding, conv2_padding,
        )
        flat_dim = _flatten_dim(
            input_time, input_channels,
            conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,
            pool_kernel, se_reduction, conv1_padding, conv2_padding,
        )
        self.flatten_dim = flat_dim
        if strict_flatten and flat_dim != target_flatten:
            self.proj = nn.Linear(flat_dim, target_flatten)
            self.flatten_dim = target_flatten
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) -> (B, flatten_dim)
        """
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.frontend(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PaperCNNSEFrontend":
        ck1 = cfg.get("conv1_kernel", [10, 10])
        ck2 = cfg.get("conv2_kernel", [15, 15])
        pk = cfg.get("pool_kernel", [10, 1])
        p1 = cfg.get("conv1_padding")
        p2 = cfg.get("conv2_padding")
        return cls(
            input_time=cfg.get("input_time", 251),
            input_channels=cfg.get("input_channels", 15),
            conv1_filters=cfg.get("conv1_filters", 16),
            conv1_kernel=(int(ck1[0]), int(ck1[1])),
            conv2_filters=cfg.get("conv2_filters", 12),
            conv2_kernel=(int(ck2[0]), int(ck2[1])),
            pool_kernel=(int(pk[0]), int(pk[1])),
            se_reduction=cfg.get("se_reduction", 3),
            conv1_padding=tuple(p1) if p1 else None,
            conv2_padding=tuple(p2) if p2 else None,
            target_flatten=cfg.get("flatten_size", PAPER_FLATTEN_SIZE),
            strict_flatten=cfg.get("strict_flatten", True),
        )
