"""EEG filtering utilities."""

from __future__ import annotations

import mne
import numpy as np


def apply_bandpass(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    verbose: bool = False,
) -> np.ndarray:
    """Apply bandpass filter to (n_channels, n_times) data."""
    return mne.filter.filter_data(
        data.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose="ERROR" if not verbose else "INFO",
    )


def filter_raw(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
) -> mne.io.BaseRaw:
    """Filter MNE Raw in place."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    return raw
