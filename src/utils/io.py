"""I/O helpers for trials, metrics, and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dict as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_trials_npz(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    session_ids: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save processed trials to NPZ."""
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        trial_ids=trial_ids,
        session_ids=session_ids,
        channel_names=np.array(channel_names, dtype=object),
        sfreq=np.array([sfreq]),
        metadata=json.dumps(metadata or {}),
    )


def load_trials_npz(path: str | Path) -> dict[str, Any]:
    """Load processed trials from NPZ."""
    data = np.load(path, allow_pickle=True)
    meta_str = data["metadata"].item() if "metadata" in data else "{}"
    if isinstance(meta_str, bytes):
        meta_str = meta_str.decode()
    metadata = json.loads(meta_str) if meta_str else {}
    channel_names = list(data["channel_names"])
    sfreq = float(data["sfreq"][0])
    return {
        "X": data["X"],
        "y": data["y"],
        "trial_ids": data["trial_ids"],
        "session_ids": data["session_ids"],
        "channel_names": channel_names,
        "sfreq": sfreq,
        "metadata": metadata,
    }


def save_metrics_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save metrics DataFrame."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def append_or_create_metrics_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Append rows to metrics CSV or create new."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        save_metrics_csv(df, path)
