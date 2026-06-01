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

# BNCI2014_001 / MOABB event names (4-class MI).
EVENT_NAMES_4 = ["left_hand", "right_hand", "feet", "tongue"]
EVENT_NAMES_2 = ["left_hand", "right_hand"]

# MOABB / MNE may return numeric event codes instead of names.
EVENT_CODE_TO_NAME: dict[int, str] = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}


def paradigm_event_names(n_classes: int) -> list[str]:
    if n_classes == 2:
        return list(EVENT_NAMES_2)
    if n_classes == 4:
        return list(EVENT_NAMES_4)
    raise ValueError(f"Unsupported n_classes={n_classes}; use 2 or 4 for BNCI2014_001")


def _normalize_event_label(value: Any) -> str:
    """Map MOABB label variants to canonical event names."""
    if isinstance(value, (int, np.integer)):
        name = EVENT_CODE_TO_NAME.get(int(value))
        if name:
            return name
        return str(int(value))
    s = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "left": "left_hand",
        "right": "right_hand",
        "both_feet": "feet",
        "foot": "feet",
        "feet": "feet",
        "tongue": "tongue",
    }
    return aliases.get(s, s)


def encode_and_filter_labels(
    y: Any,
    event_names: list[str],
    X: np.ndarray | None = None,
    meta: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any | None]:
    """
    Convert MOABB labels (strings or event codes) to 0..K-1 integers.

    Drops trials whose label is not in ``event_names`` (e.g. feet/tongue when K=2).
    """
    y_raw = np.asarray(y)
    name_to_id = {name: i for i, name in enumerate(event_names)}
    encoded = np.full(len(y_raw), -1, dtype=np.int64)
    for i, val in enumerate(y_raw):
        canonical = _normalize_event_label(val)
        if canonical in name_to_id:
            encoded[i] = name_to_id[canonical]

    keep = encoded >= 0
    if not np.all(keep):
        dropped = int((~keep).sum())
        unique_bad = sorted(
            {_normalize_event_label(v) for v in y_raw[~keep]}
        )
        logger.warning(
            "Dropping %d trials with labels outside %s: %s",
            dropped,
            event_names,
            unique_bad,
        )
        encoded = encoded[keep]
        if X is not None:
            X = X[keep]
        if meta is not None and hasattr(meta, "iloc"):
            meta = meta.loc[keep].reset_index(drop=True)

    return encoded.astype(np.int64), keep, X, meta


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
    event_names = paradigm_event_names(n_classes)
    paradigm_kw: dict[str, Any] = {
        "n_classes": n_classes,
        "fmin": fmin,
        "fmax": fmax,
        "tmin": tmin,
        "tmax": tmax,
        "resample": resample,
    }
    try:
        paradigm = MotorImagery(events=event_names, **paradigm_kw)
    except TypeError:
        # Older MOABB: no ``events`` kwarg — filter labels after get_data.
        logger.warning("MOABB MotorImagery without events=; filtering labels manually.")
        paradigm = MotorImagery(**paradigm_kw)

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

        y, keep, X, meta = encode_and_filter_labels(y, event_names, X=X, meta=meta)
        if len(y) == 0:
            raise ValueError(
                f"No trials left for {subject_id} after filtering to {event_names}. "
                "Check MOABB version and MotorImagery events."
            )

        X = _pad_or_crop_time(X.astype(np.float32), expected_samples)
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
                "event_names": event_names,
                "label_map": {name: i for i, name in enumerate(event_names)},
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
