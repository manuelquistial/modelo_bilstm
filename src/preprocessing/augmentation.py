"""Overlapping window segmentation (data augmentation)."""

from __future__ import annotations

import numpy as np


def n_windows_from_length(
    signal_length: int,
    window_size: int,
    step_size: int,
) -> int:
    """Number of overlapping windows."""
    if signal_length < window_size:
        raise ValueError(f"signal_length {signal_length} < window_size {window_size}")
    return (signal_length - window_size) // step_size + 1


def create_overlapping_windows(
    X_trials: np.ndarray,
    y_trials: np.ndarray,
    trial_ids: np.ndarray,
    window_size: int = 251,
    step_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation per trial.

    Parameters
    ----------
    X_trials : (n_trials, n_time, n_channels)
    y_trials : (n_trials,)
    trial_ids : (n_trials,)

    Returns
    -------
    X_segments, y_segments, segment_trial_ids, segment_start_indices, segment_end_indices
    """
    n_trials, n_time, n_channels = X_trials.shape
    n_win = n_windows_from_length(n_time, window_size, step_size)

    segments: list[np.ndarray] = []
    y_seg: list[int] = []
    tid_seg: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    for i in range(n_trials):
        trial = X_trials[i]
        for w in range(n_win):
            start = w * step_size
            end = start + window_size
            segments.append(trial[start:end, :])
            y_seg.append(int(y_trials[i]))
            tid_seg.append(int(trial_ids[i]))
            starts.append(start)
            ends.append(end)

    X_segments = np.stack(segments, axis=0)
    return (
        X_segments.astype(np.float32),
        np.array(y_seg, dtype=np.int64),
        np.array(tid_seg, dtype=np.int64),
        np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64),
    )
