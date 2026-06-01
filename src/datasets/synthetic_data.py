"""Synthetic EEG data generator for pipeline testing without hardware."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import CHANNEL_NAMES, RAW_SFREQ, TRIAL_DURATION_SEC
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

C3_IDX = CHANNEL_NAMES.index("C3")
C4_IDX = CHANNEL_NAMES.index("C4")


def _generate_trial_eeg(
    label: int,
    n_samples: int,
    n_channels: int,
    sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one trial (n_channels, n_samples) with class-discriminative C3/C4."""
    t = np.arange(n_samples) / sfreq
    data = rng.normal(0, 5, size=(n_channels, n_samples))

    # Mu rhythm modulation during MI (2-6 s)
    mi_start = int(2.0 * sfreq)
    mi_end = int(6.0 * sfreq)
    mi_t = t[mi_start:mi_end]

    if label == 0:  # left thigh -> ERD at C4, ERS at C3
        data[C3_IDX, mi_start:mi_end] += 3 * np.sin(2 * np.pi * 10 * mi_t)
        data[C4_IDX, mi_start:mi_end] -= 2 * np.sin(2 * np.pi * 10 * mi_t)
        data[C3_IDX, mi_start:mi_end] += 1.5 * np.sin(2 * np.pi * 20 * mi_t)
    else:  # right -> opposite
        data[C4_IDX, mi_start:mi_end] += 3 * np.sin(2 * np.pi * 10 * mi_t)
        data[C3_IDX, mi_start:mi_end] -= 2 * np.sin(2 * np.pi * 10 * mi_t)
        data[C4_IDX, mi_start:mi_end] += 1.5 * np.sin(2 * np.pi * 20 * mi_t)

    return data.astype(np.float64)


def generate_session_events(
    subject_id: str,
    session_id: int,
    n_trials: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate balanced trial schedule with paired random blocks."""
    rng = np.random.default_rng(seed + session_id)
    labels: list[int] = []
    while len(labels) < n_trials:
        first = int(rng.integers(0, 2))
        labels.extend([first, 1 - first])
    labels = labels[:n_trials]

    rows = []
    t0 = 0.0
    for tid, lab in enumerate(labels, start=1):
        trial_start = t0
        prep_start = trial_start
        cue_start = trial_start + 2.0
        cue_end = cue_start + 4.0
        rest_start = cue_end
        trial_end = trial_start + TRIAL_DURATION_SEC
        rows.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "trial_id": tid + (session_id - 1) * n_trials,
            "label": lab,
            "cue": "left" if lab == 0 else "right",
            "trial_start": trial_start,
            "cue_start": cue_start,
            "cue_end": cue_end,
            "rest_start": rest_start,
            "trial_end": trial_end,
        })
        t0 = trial_end + 1.0  # 1 s inter-trial interval
    return pd.DataFrame(rows)


def generate_subject(
    subject_id: str,
    output_dir: Path,
    n_sessions: int = 3,
    trials_per_session: int = 50,
    sfreq: float = RAW_SFREQ,
    seed: int = 42,
) -> None:
    """Generate raw CSV data for one subject."""
    rng = np.random.default_rng(seed + int(subject_id.replace("S", "")))
    n_channels = len(CHANNEL_NAMES)
    samples_per_trial = int(TRIAL_DURATION_SEC * sfreq)

    for sess in range(1, n_sessions + 1):
        sess_dir = ensure_dir(output_dir / subject_id / f"session_{sess:02d}")
        events = generate_session_events(subject_id, sess, trials_per_session, seed)

        eeg_chunks: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        global_t = 0.0

        for _, row in events.iterrows():
            label = int(row["label"])
            trial_eeg = _generate_trial_eeg(
                label, samples_per_trial, n_channels, sfreq, rng
            )
            n_samp = trial_eeg.shape[1]
            ts = global_t + np.arange(n_samp) / sfreq
            eeg_chunks.append(trial_eeg.T)
            timestamps.append(ts)
            global_t = float(ts[-1]) + 1.0 / sfreq + 1.0

        eeg_matrix = np.vstack(eeg_chunks)
        ts_all = np.concatenate(timestamps)
        df_eeg = pd.DataFrame(eeg_matrix, columns=CHANNEL_NAMES)
        df_eeg.insert(0, "timestamp", ts_all)
        df_eeg.to_csv(sess_dir / "eeg_raw.csv", index=False)
        events.to_csv(sess_dir / "events.csv", index=False)
        logger.info("Generated %s session %d: %d trials", subject_id, sess, len(events))


def generate_all_subjects(
    output_dir: Path,
    subjects: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Generate synthetic data for all subjects."""
    subjects = subjects or [f"S{i:02d}" for i in range(1, 11)]
    for sid in subjects:
        subj_seed = seed + int(sid.replace("S", ""))
        generate_subject(sid, output_dir, seed=subj_seed)
    logger.info("Synthetic data written to %s", output_dir)
