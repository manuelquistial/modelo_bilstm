"""End-to-end preprocessing pipeline for one subject."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.artifact_removal import run_ica
from src.preprocessing.epoching import (
    build_mne_raw,
    extract_mi_epochs,
    load_eeg_matrix_from_csv,
)
from src.preprocessing.filters import filter_raw
from src.preprocessing.quality_checks import check_epoch_shape, check_labels_binary
from src.preprocessing.resampling import resample_raw
from src.utils.constants import CHANNEL_NAMES, MONTAGE_RENAME_MAP
from src.utils.io import save_json, save_trials_npz

logger = logging.getLogger(__name__)


def load_raw_subject_session(
    subject_dir: Path,
    channel_names: list[str] | None = None,
) -> tuple[list[np.ndarray], list[pd.DataFrame], list[int], list[float]]:
    """Load all sessions for a subject from raw CSV files."""
    channel_names = channel_names or CHANNEL_NAMES
    eeg_list: list[np.ndarray] = []
    events_list: list[pd.DataFrame] = []
    session_ids: list[int] = []
    time_offsets: list[float] = []

    sessions = sorted(subject_dir.glob("session_*"))
    for sess_path in sessions:
        eeg_path = sess_path / "eeg_raw.csv"
        ev_path = sess_path / "events.csv"
        if not eeg_path.exists() or not ev_path.exists():
            logger.warning("Skipping incomplete session: %s", sess_path)
            continue
        eeg, t_off = load_eeg_matrix_from_csv(eeg_path, channel_names)
        eeg_list.append(eeg)
        events_list.append(pd.read_csv(ev_path))
        time_offsets.append(t_off)
        sid = int(sess_path.name.split("_")[-1])
        session_ids.append(sid)
    return eeg_list, events_list, session_ids, time_offsets


def preprocess_session(
    eeg: np.ndarray,
    events_df: pd.DataFrame,
    config: dict[str, Any],
    channel_names: list[str] | None = None,
    time_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess one session."""
    channel_names = channel_names or CHANNEL_NAMES
    raw_sfreq = config.get("raw_sfreq", 1000)
    target_sfreq = config.get("target_sfreq", 125)

    raw = build_mne_raw(eeg, channel_names, raw_sfreq, MONTAGE_RENAME_MAP)
    resample_raw(raw, target_sfreq)
    filter_raw(raw, config.get("highpass", 1.0), config.get("lowpass", 40.0))
    if config.get("ica_enabled", True):
        run_ica(
            raw,
            n_components=config.get("ica_n_components", 0.95),
            random_state=config.get("ica_random_state", 42),
            max_iter=config.get("ica_max_iter", 800),
        )
    X, y, trial_ids = extract_mi_epochs(
        raw,
        events_df,
        tmin=config.get("epoch_tmin", 2.0),
        tmax=config.get("epoch_tmax", 6.0),
        expected_samples=config.get("expected_epoch_samples", 501),
        time_offset=time_offset,
    )
    return X, y, trial_ids


def preprocess_subject(
    subject_id: str,
    input_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Preprocess all sessions for one subject and save trials.npz."""
    channel_names = channel_names or CHANNEL_NAMES
    subject_path = input_dir / subject_id
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_path}")

    eeg_list, events_list, session_nums, time_offsets = load_raw_subject_session(
        subject_path, channel_names
    )
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_tids: list[np.ndarray] = []
    all_sess: list[np.ndarray] = []

    for eeg, ev, sid, t_off in zip(eeg_list, events_list, session_nums, time_offsets):
        X, y, tids = preprocess_session(eeg, ev, config, channel_names, time_offset=t_off)
        all_X.append(X)
        all_y.append(y)
        all_tids.append(tids)
        all_sess.append(np.full(len(y), sid, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    trial_ids = np.concatenate(all_tids, axis=0)
    session_ids = np.concatenate(all_sess, axis=0)

    check_epoch_shape(X, config.get("expected_epoch_samples", 501), len(channel_names))
    check_labels_binary(y)

    out_subj = output_dir / subject_id
    out_subj.mkdir(parents=True, exist_ok=True)
    metadata = {
        "subject_id": subject_id,
        "n_trials": int(len(y)),
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    save_trials_npz(
        out_subj / "trials.npz",
        X,
        y,
        trial_ids,
        session_ids,
        channel_names,
        config.get("target_sfreq", 125),
        metadata,
    )
    save_json(metadata, out_subj / "metadata.json")
    logger.info("Saved processed trials for %s: X=%s", subject_id, X.shape)
    return {"X": X, "y": y, "trial_ids": trial_ids, "session_ids": session_ids, "metadata": metadata}
