"""BrainFlow EEG streaming for OpenBCI."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.acquisition.openbci_config import OpenBCIConfig

logger = logging.getLogger(__name__)


class BrainFlowStream:
    """Wrapper around BrainFlow BoardShim."""

    def __init__(self, config: OpenBCIConfig, mock: bool = False) -> None:
        self.config = config
        self.mock = mock
        self.board = None
        self._mock_data: np.ndarray | None = None

    def prepare_session(self) -> None:
        if self.mock:
            logger.info("BrainFlow mock mode — no hardware connection")
            return
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as exc:
            raise ImportError("brainflow required for real acquisition") from exc

        params = BrainFlowInputParams()
        if self.config.serial_port:
            params.serial_port = self.config.serial_port
        if self.config.ip_address:
            params.ip_address = self.config.ip_address
            params.ip_port = self.config.ip_port

        self.board = BoardShim(self.config.board_id, params)
        self.board.prepare_session()
        logger.info("BrainFlow session prepared board_id=%s", self.config.board_id)

    def start_stream(self) -> None:
        if self.mock:
            return
        if self.board is None:
            raise RuntimeError("Call prepare_session first")
        self.board.start_stream()
        logger.info("Stream started")

    def stop_stream(self) -> None:
        if self.mock or self.board is None:
            return
        self.board.stop_stream()

    def release(self) -> None:
        if self.mock or self.board is None:
            return
        self.board.release_session()

    def get_board_data(self) -> np.ndarray:
        """Return EEG data (n_channels, n_samples)."""
        if self.mock:
            n = self.config.sampling_rate * 10
            return np.random.randn(len(self.config.channel_names), n) * 5
        if self.board is None:
            raise RuntimeError("Board not initialized")
        data = self.board.get_board_data()
        eeg_channels = self.board.get_eeg_channels(self.config.board_id)
        return data[eeg_channels, :]

    def stream_for_duration(self, duration_sec: float, poll_hz: float = 100.0) -> np.ndarray:
        """Collect data for duration_sec."""
        chunks = []
        n_polls = int(duration_sec * poll_hz)
        for _ in range(n_polls):
            chunks.append(self.get_board_data())
            time.sleep(1.0 / poll_hz)
        if not chunks:
            return np.zeros((len(self.config.channel_names), 0))
        return np.hstack(chunks)
