"""Epoch extraction from continuous EEG."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd


def extract_mi_epochs(
    raw: mne.io.BaseRaw,
    events_df: pd.DataFrame,
    tmin: float = 2.0,
    tmax: float = 6.0,
    expected_samples: int = 501,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract motor imagery epochs (2-6 s) per trial.

    Returns X (n_trials, n_samples, n_channels), y, trial_ids.
    """
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    n_channels = len(ch_names)
    epochs_list: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []

    data = raw.get_data()  # (n_ch, n_times)
    times = raw.times

    for _, row in events_df.iterrows():
        cue_start = float(row["cue_start"])
        label = int(row["label"])
        tid = int(row["trial_id"])
        start_idx = int(np.argmin(np.abs(times - cue_start)))
        end_time = cue_start + (tmax - tmin)
        end_idx = int(np.argmin(np.abs(times - end_time)))
        segment = data[:, start_idx:end_idx]
        if segment.shape[1] < expected_samples:
            pad = expected_samples - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad)), mode="edge")
        elif segment.shape[1] > expected_samples:
            segment = segment[:, :expected_samples]
        epochs_list.append(segment.T)  # (time, channels)
        labels.append(label)
        trial_ids.append(tid)

    X = np.stack(epochs_list, axis=0)
    y = np.array(labels, dtype=np.int64)
    tids = np.array(trial_ids, dtype=np.int64)
    return X, y, tids


def build_mne_raw(
    eeg_data: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    montage_map: dict[str, str] | None = None,
) -> mne.io.RawArray:
    """Build MNE RawArray from (n_channels, n_times)."""
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data.astype(np.float64), info, verbose="ERROR")
    montage_names = channel_names
    if montage_map:
        montage_names = [montage_map.get(c, c) for c in channel_names]
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw
