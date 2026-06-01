"""Import BNCI2014_001 (MOABB) into project trials.npz format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir, save_json, save_trials_npz

logger = logging.getLogger(__name__)

# Standard 22 EEG channels for BNCI Horizon 2014-001 (Guger montage).
BNCI2014_001_CHANNEL_NAMES: list[str] = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

DATASET_ID = "BNCI2014_001"


def _pad_or_crop_time(X: np.ndarray, target_samples: int) -> np.ndarray:
    """Ensure last axis (time) has length target_samples."""
    n_time = X.shape[1]
    if n_time == target_samples:
        return X
    if n_time > target_samples:
        start = (n_time - target_samples) // 2
        return X[:, start : start + target_samples, :]
    pad_total = target_samples - n_time
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(
        X,
        ((0, 0), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


def import_bnci2014_001(
    output_dir: str | Path,
    subjects: list[int] | None = None,
    n_classes: int = 2,
    tmin: float = 0.0,
    tmax: float = 4.0,
    resample: float = 125.0,
    expected_samples: int = 501,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> dict[str, Any]:
    """
    Download (if needed) and import BNCI2014_001 via MOABB MotorImagery paradigm.

    Saves per subject: ``{output_dir}/S{subj:02d}/trials.npz`` with
    X shape (n_trials, expected_samples, n_channels).

    Returns summary dict with per-subject trial counts.
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError(
            "MOABB is required for BNCI2014_001. Install with: pip install moabb"
        ) from exc

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        resample=resample,
    )

    subject_list = subjects if subjects is not None else list(dataset.subject_list)
    summary: dict[str, Any] = {
        "dataset": DATASET_ID,
        "n_classes": n_classes,
        "tmin": tmin,
        "tmax": tmax,
        "resample_hz": resample,
        "expected_samples": expected_samples,
        "subjects": {},
    }

    for subj in subject_list:
        subject_id = f"S{int(subj):02d}"
        logger.info("Loading %s subject %s …", DATASET_ID, subject_id)

        X, y, meta = paradigm.get_data(dataset, subjects=[int(subj)])
        # MOABB: (n_trials, n_channels, n_times)
        if X.ndim != 3:
            raise ValueError(f"Unexpected X shape for {subject_id}: {X.shape}")

        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))  # -> (n_trials, n_times, n_channels)
        else:
            # Already (n_trials, n_times, n_channels) in some versions
            if X.shape[2] not in (len(BNCI2014_001_CHANNEL_NAMES), 22):
                X = np.transpose(X, (0, 2, 1))

        X = _pad_or_crop_time(X.astype(np.float32), expected_samples)
        y = np.asarray(y, dtype=np.int64)
        n_trials = len(y)
        trial_ids = np.arange(n_trials, dtype=np.int64)

        if "session" in meta.columns:
            session_ids = meta["session"].astype(str).to_numpy()
        else:
            session_ids = np.array(["session_01"] * n_trials, dtype=object)

        ch_names = list(BNCI2014_001_CHANNEL_NAMES)
        if X.shape[2] != len(ch_names):
            ch_names = [f"Ch{i}" for i in range(X.shape[2])]
            logger.warning(
                "%s: expected %d channels, got %d — using generic names",
                subject_id,
                len(BNCI2014_001_CHANNEL_NAMES),
                X.shape[2],
            )

        out_subj = output_dir / subject_id
        ensure_dir(out_subj)
        save_trials_npz(
            out_subj / "trials.npz",
            X=X,
            y=y,
            trial_ids=trial_ids,
            session_ids=session_ids,
            channel_names=ch_names,
            sfreq=resample,
            metadata={
                "source": DATASET_ID,
                "moabb_subject": int(subj),
                "n_classes": n_classes,
                "paradigm": "MotorImagery",
                "tmin": tmin,
                "tmax": tmax,
            },
        )
        save_json(
            {
                "subject_id": subject_id,
                "dataset": DATASET_ID,
                "n_trials": int(n_trials),
                "class_counts": {
                    str(int(c)): int((y == c).sum()) for c in np.unique(y)
                },
                "X_shape": list(X.shape),
            },
            out_subj / "import_meta.json",
        )
        summary["subjects"][subject_id] = {
            "n_trials": int(n_trials),
            "X_shape": list(X.shape),
            "class_counts": {str(int(c)): int((y == c).sum()) for c in np.unique(y)},
        }
        logger.info(
            "%s: saved %d trials, shape %s",
            subject_id,
            n_trials,
            X.shape,
        )

    save_json(summary, output_dir / "bnci2014_001_import_summary.json")
    return summary
