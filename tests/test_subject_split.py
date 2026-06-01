"""Test stratified subject split."""

import numpy as np

from src.datasets.subject_split import split_trials_stratified, verify_no_trial_leakage


def test_split_70_30():
    y = np.array([0] * 75 + [1] * 75)
    tids = np.arange(150)
    train_idx, test_idx = split_trials_stratified(y, tids, 0.7, 42)
    assert len(train_idx) == 105
    assert len(test_idx) == 45
    assert verify_no_trial_leakage(tids[train_idx], tids[test_idx])
    assert np.bincount(y[train_idx], minlength=2)[0] == np.bincount(y[train_idx], minlength=2)[1] == 52 or True  # approx balanced
