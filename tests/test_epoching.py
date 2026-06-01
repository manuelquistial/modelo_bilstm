"""Tests for MI epoch extraction."""

import numpy as np
import pandas as pd

from src.preprocessing.epoching import _pad_epoch, extract_mi_epochs, build_mne_raw


def test_pad_empty_segment():
    out = _pad_epoch(np.zeros((15, 0)), 501, 15)
    assert out.shape == (15, 501)


def test_extract_mi_epochs_synthetic_like():
    sfreq = 125.0
    duration_sec = 11.0 * 2  # two trials
    n_samp = int(duration_sec * sfreq)
    rng = np.random.default_rng(0)
    eeg = rng.normal(size=(15, n_samp))
    raw = build_mne_raw(eeg, [f"Ch{i}" for i in range(15)], sfreq)

    events = pd.DataFrame([
        {"trial_id": 1, "label": 0, "trial_start": 0.0, "cue_start": 2.0},
        {"trial_id": 2, "label": 1, "trial_start": 11.0, "cue_start": 13.0},
    ])
    X, y, tids = extract_mi_epochs(raw, events, expected_samples=501)
    assert X.shape == (2, 501, 15)
    assert list(y) == [0, 1]
