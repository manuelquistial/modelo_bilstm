"""Epoch extraction from continuous EEG."""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _epoch_window_indices(
    times: np.ndarray,
    sfreq: float,
    trial_start: float,
    tmin: float,
    tmax: float,
    expected_samples: int,
) -> tuple[int, int]:
    """
    Sample indices for MI window (tmin–tmax s relative to trial_start).

    Uses rounded sample indices and clamps to recording bounds.
    """
    t_origin = float(times[0]) if len(times) else 0.0
    epoch_start = trial_start + tmin
    epoch_end = trial_start + tmax

    start_idx = int(np.round((epoch_start - t_origin) * sfreq))
    end_idx = int(np.round((epoch_end - t_origin) * sfreq))

    n_times = len(times)
    if n_times == 0:
        return 0, 0

    start_idx = max(0, min(start_idx, n_times - 1))
    end_idx = max(start_idx + 1, min(end_idx, n_times))

    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, n_times)

    return start_idx, end_idx


def _pad_epoch(segment: np.ndarray, expected_samples: int, n_channels: int) -> np.ndarray:
    """Pad or crop channel x time array to (n_channels, expected_samples)."""
    if segment.ndim != 2:
        raise ValueError(f"Expected 2D segment, got shape {segment.shape}")

    n_ch, n_samp = segment.shape
    if n_ch != n_channels and n_samp == n_channels:
        segment = segment.T
        n_ch, n_samp = segment.shape

    if n_samp == 0:
        logger.warning("Empty MI segment; filling with zeros.")
        return np.zeros((n_channels, expected_samples), dtype=np.float64)

    if n_samp < expected_samples:
        pad = expected_samples - n_samp
        segment = np.pad(segment, ((0, 0), (0, pad)), mode="edge")
    elif n_samp > expected_samples:
        segment = segment[:, :expected_samples]
    return segment


def extract_mi_epochs(
    raw: mne.io.BaseRaw,
    events_df: pd.DataFrame,
    tmin: float = 2.0,
    tmax: float = 6.0,
    expected_samples: int = 501,
    time_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract motor imagery epochs (2–6 s from trial onset) per trial.

    Parameters
    ----------
    time_offset : float
        Subtract from event times when CSV timestamps do not start at 0.

    Returns
    -------
    X : (n_trials, expected_samples, n_channels)
    y, trial_ids
    """
    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    data = raw.get_data()
    times = raw.times

    if data.shape[1] == 0:
        raise ValueError("Raw EEG has zero time samples; cannot extract epochs.")

    epochs_list: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []

    for _, row in events_df.iterrows():
        label = int(row["label"])
        tid = int(row["trial_id"])

        if "trial_start" in row and pd.notna(row["trial_start"]):
            trial_start = float(row["trial_start"]) - time_offset
        elif "cue_start" in row and pd.notna(row["cue_start"]):
            trial_start = float(row["cue_start"]) - time_offset - tmin
        else:
            raise KeyError("events.csv must contain trial_start or cue_start")

        start_idx, end_idx = _epoch_window_indices(
            times, sfreq, trial_start, tmin, tmax, expected_samples
        )
        segment = data[:, start_idx:end_idx]
        segment = _pad_epoch(segment, expected_samples, n_channels)
        epochs_list.append(segment.T.astype(np.float32))

        labels.append(label)
        trial_ids.append(tid)

    if not epochs_list:
        raise ValueError("No epochs extracted; check events.csv and eeg_raw.csv alignment.")

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
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def load_eeg_matrix_from_csv(
    eeg_path: str | pd.PathLike,
    channel_names: list[str],
) -> tuple[np.ndarray, float]:
    """
    Load EEG as (n_channels, n_times) and time offset from first timestamp.

    Returns
    -------
    eeg_data, time_offset (subtract from event times for alignment)
    """
    df = pd.read_csv(eeg_path)
    time_offset = 0.0
    if "timestamp" in df.columns:
        time_offset = float(df["timestamp"].iloc[0])
    ch_cols = [c for c in channel_names if c in df.columns]
    if not ch_cols:
        raise ValueError(f"No channel columns found in {eeg_path}")
    eeg = df[ch_cols].values.T.astype(np.float64)
    return eeg, time_offset
