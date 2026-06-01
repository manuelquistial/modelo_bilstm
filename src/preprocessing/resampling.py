"""Resampling utilities."""

from __future__ import annotations

import mne


def resample_raw(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    """Downsample or upsample MNE Raw."""
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose="ERROR")
    return raw
