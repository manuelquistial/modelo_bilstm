"""OpenBCI / BrainFlow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OpenBCIConfig:
    board_id: int
    serial_port: str
    ip_address: str
    ip_port: int
    sampling_rate: int
    n_channels: int
    channel_names: list[str]
    output_dir: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OpenBCIConfig":
        return cls(
            board_id=int(cfg.get("board_id", 0)),
            serial_port=str(cfg.get("serial_port", "")),
            ip_address=str(cfg.get("ip_address", "192.168.4.1")),
            ip_port=int(cfg.get("ip_port", 6677)),
            sampling_rate=int(cfg.get("sampling_rate", 1000)),
            n_channels=int(cfg.get("n_channels", 16)),
            channel_names=list(cfg.get("channel_names", [])),
            output_dir=str(cfg.get("output_dir", "data/raw")),
        )
