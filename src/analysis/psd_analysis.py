"""Power spectral density analysis in mu and beta bands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_welch

from src.utils.constants import CHANNEL_NAMES, MONTAGE_RENAME_MAP
from src.utils.io import load_trials_npz

logger = logging.getLogger(__name__)


def compute_psd_trials(
    X: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """PSD for trials X (n_trials, n_time, n_channels). Returns freqs, psd (n, ch, freq)."""
    X_mne = np.transpose(X, (0, 2, 1))
    psds, freqs = psd_array_welch(X_mne, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return freqs, psds


def band_mean_power(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Mean power in band, averaged over freq -> (n_trials, n_channels)."""
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return psd[..., idx].mean(axis=-1)


def run_psd_subject(
    subject_id: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Generate PSD plots and topomaps per class."""
    data = load_trials_npz(processed_dir / subject_id / "trials.npz")
    X, y = data["X"], data["y"]
    sfreq = data["sfreq"]
    ch_names = data["channel_names"]
    bands = config.get("bands", {"mu": [8, 12], "beta": [13, 30]})

    freqs, psd = compute_psd_trials(X, sfreq)
    output_dir.mkdir(parents=True, exist_ok=True)

    montage_names = [MONTAGE_RENAME_MAP.get(c, c) for c in ch_names]
    info = mne.create_info(montage_names, sfreq, "eeg")
    try:
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for cls, name in [(0, "left"), (1, "right")]:
        mask = y == cls
        for bi, (bname, band) in enumerate(bands.items()):
            bp = band_mean_power(psd[mask], freqs, tuple(band)).mean(axis=0)
            ax = axes[bi, cls]
            try:
                mne.viz.plot_topomap(bp, info, axes=ax, show=False)
            except Exception:
                ax.bar(range(len(bp)), bp)
                ax.set_xticks(range(len(ch_names)))
                ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            ax.set_title(f"{name} — {bname}")
    fig.suptitle(f"PSD topomaps {subject_id}")
    fig.savefig(output_dir / f"psd_topomap_subject_{subject_id}_mu_beta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PSD analysis for %s", subject_id)
