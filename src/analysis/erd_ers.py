"""ERD/ERS analysis for C3 and C4."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

from src.preprocessing.epoching import build_mne_raw
from src.preprocessing.preprocessing_pipeline import load_raw_subject_session
from src.preprocessing.filters import filter_raw
from src.preprocessing.resampling import resample_raw
from src.utils.constants import CHANNEL_NAMES

logger = logging.getLogger(__name__)


def band_power(sig: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, len(sig)))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return float(trapezoid(psd[idx], freqs[idx]))


def erd_percent(power_task: float, power_baseline: float) -> float:
    if power_baseline < 1e-12:
        return 0.0
    return ((power_task - power_baseline) / power_baseline) * 100.0


def run_erd_subject(
    subject_id: str,
    raw_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    preprocess_cfg: dict[str, Any],
) -> None:
    """Compute ERD/ERS for C3/C4 from continuous preprocessed data."""
    subject_path = raw_dir / subject_id
    if not subject_path.exists():
        logger.warning("Raw data missing for ERD: %s", subject_id)
        return

    eeg_list, events_list, _ = load_raw_subject_session(subject_path)
    sfreq_target = preprocess_cfg.get("target_sfreq", 125)
    bands = config.get("bands", {"mu": [8, 12], "beta": [13, 30]})
    erd_chs = config.get("erd_channels", ["C3", "C4"])
    baseline_win = config.get("erd_baseline", [0, 2])
    task_win = config.get("erd_task", [2, 6])

    rows = []
    curves: dict[str, list[float]] = {ch: [] for ch in erd_chs}

    for eeg, ev in zip(eeg_list, events_list):
        raw = build_mne_raw(eeg, CHANNEL_NAMES, preprocess_cfg.get("raw_sfreq", 1000))
        resample_raw(raw, sfreq_target)
        filter_raw(raw, preprocess_cfg.get("highpass", 1), preprocess_cfg.get("lowpass", 40))
        data = raw.get_data()
        times = raw.times
        ch_idx = {ch: CHANNEL_NAMES.index(ch) for ch in erd_chs}

        for _, row in ev.iterrows():
            label = int(row["label"])
            t0 = float(row["trial_start"])
            b0, b1 = t0 + baseline_win[0], t0 + baseline_win[1]
            k0, k1 = t0 + task_win[0], t0 + task_win[1]
            for ch, idx in ch_idx.items():
                for bname, band in bands.items():
                    i0 = int(np.argmin(np.abs(times - b0)))
                    i1 = int(np.argmin(np.abs(times - b1)))
                    k2 = int(np.argmin(np.abs(times - k0)))
                    k3 = int(np.argmin(np.abs(times - k1)))
                    p_base = band_power(data[idx, i0:i1], sfreq_target, tuple(band))
                    p_task = band_power(data[idx, k2:k3], sfreq_target, tuple(band))
                    erd = erd_percent(p_task, p_base)
                    rows.append({
                        "subject_id": subject_id,
                        "class": label,
                        "channel": ch,
                        "band": bname,
                        "erd_percent": erd,
                    })
                    curves[ch].append(erd)

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"erd_ers_subject_{subject_id}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for ch in erd_chs:
        sub = df[df["channel"] == ch]
        for cls, color in [(0, "C0"), (1, "C1")]:
            vals = sub[sub["class"] == cls]["erd_percent"].values
            if len(vals):
                ax.plot(vals.cumsum() / np.arange(1, len(vals) + 1), label=f"{ch} class {cls}", color=color, alpha=0.7)
    ax.set_xlabel("Trial index (running mean)")
    ax.set_ylabel("ERD %")
    ax.set_title(f"ERD/ERS {subject_id} C3/C4")
    ax.legend()
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    fig.savefig(output_dir / f"erd_ers_subject_{subject_id}_C3_C4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ERD/ERS for %s", subject_id)
