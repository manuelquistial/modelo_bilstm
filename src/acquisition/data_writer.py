"""Save raw EEG and event files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_eeg_csv(
    path: Path,
    timestamps: np.ndarray,
    eeg: np.ndarray,
    channel_names: list[str],
) -> None:
    """Save eeg_raw.csv with timestamp + channels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(eeg.T, columns=channel_names)
    df.insert(0, "timestamp", timestamps)
    df.to_csv(path, index=False)


def append_eeg_csv(
    path: Path,
    timestamps: np.ndarray,
    eeg: np.ndarray,
    channel_names: list[str],
) -> None:
    df = pd.DataFrame(eeg.T, columns=channel_names)
    df.insert(0, "timestamp", timestamps)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        save_eeg_csv(path, timestamps, eeg, channel_names)


def save_events_csv(path: Path, events: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
