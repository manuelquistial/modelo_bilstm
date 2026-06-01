"""ICA artifact removal — EEGLAB ART analogue (Sun et al. §2.3, [31])."""

from __future__ import annotations

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)

FRONTAL_CHANNELS = {"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"}


def run_ica(
    raw: mne.io.BaseRaw,
    n_components: float | int = 0.95,
    random_state: int = 42,
    max_iter: int = 500,
    exclude_heuristic: bool = True,
) -> mne.io.BaseRaw:
    """
    ICA + automatic IC rejection (EOG/EMG proxy), matching EEGLAB ART workflow.

    On failure, logs warning and returns band-passed raw unchanged.
    """
    try:
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            method="fastica",
        )
        ica.fit(raw, verbose="ERROR")
        exclude: list[int] = []
        if exclude_heuristic:
            exclude = _art_like_exclude(ica, raw)
        if exclude:
            ica.exclude = exclude
            logger.info("ICA excluding components (ART-like): %s", exclude)
        ica.apply(raw, verbose="ERROR")
    except Exception as exc:
        logger.warning("ICA failed (%s); continuing with filtered signal only.", exc)
    return raw


def _art_like_exclude(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw) -> list[int]:
    """Identify artifact ICs via frontal correlation, kurtosis, and variance."""
    sources = ica.get_sources(raw).get_data()
    n_comp = sources.shape[0]
    ch_names = raw.ch_names
    frontal_idx = [i for i, c in enumerate(ch_names) if c in FRONTAL_CHANNELS]

    mixing = ica.get_components()
    exclude: set[int] = set()

    # High correlation with frontal channels -> eye blink / EOG proxy
    for comp in range(n_comp):
        if frontal_idx:
            weights = np.abs(mixing[frontal_idx, comp])
            if np.max(weights) > 0.25 and np.mean(weights) > 0.1:
                exclude.add(comp)

    # High kurtosis / variance (muscle, spikes)
    kurt = np.abs(_kurtosis(sources, axis=1))
    var = np.var(sources, axis=1)
    scores = kurt * (var / (var.mean() + 1e-12))
    threshold = np.percentile(scores, 85)
    for i, s in enumerate(scores):
        if s >= threshold:
            exclude.add(i)

    return sorted(exclude)[: max(1, n_comp // 4)]


def _kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + 1e-12
    z = (x - mean) / std
    return np.mean(z**4, axis=axis) - 3.0
