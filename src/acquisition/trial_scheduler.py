"""Trial schedule generation for motor imagery paradigm."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_session_schedule(
    subject_id: str,
    session_id: int,
    n_trials: int = 50,
    seed: int = 42,
    trial_duration: float = 10.0,
) -> pd.DataFrame:
    """
    Paired-block randomization: blocks of 2 with opposite directions.

    Class 0 = left thigh MI, Class 1 = right thigh MI.
    """
    rng = np.random.default_rng(seed)
    labels: list[int] = []
    while len(labels) < n_trials:
        first = int(rng.integers(0, 2))
        labels.extend([first, 1 - first])
    labels = labels[:n_trials]

    rows = []
    t = 0.0
    base_tid = (session_id - 1) * n_trials
    for i, lab in enumerate(labels):
        trial_start = t
        cue_start = trial_start + 2.0
        cue_end = cue_start + 4.0
        rest_start = cue_end
        trial_end = trial_start + trial_duration
        rows.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "trial_id": base_tid + i + 1,
            "label": lab,
            "cue": "left" if lab == 0 else "right",
            "trial_start": trial_start,
            "cue_start": cue_start,
            "cue_end": cue_end,
            "rest_start": rest_start,
            "trial_end": trial_end,
        })
        t = trial_end + 1.0
    return pd.DataFrame(rows)
