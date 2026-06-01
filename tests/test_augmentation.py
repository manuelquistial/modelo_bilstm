"""Tests for overlapping window segmentation."""

import numpy as np
import pytest

from src.preprocessing.augmentation import create_overlapping_windows, n_windows_from_length


def test_six_windows_from_501_samples():
    assert n_windows_from_length(501, 251, 50) == 6


def test_segment_shapes_and_labels():
    n_trials, n_time, n_ch = 4, 501, 15
    X = np.random.randn(n_trials, n_time, n_ch).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    tids = np.array([1, 2, 3, 4])
    Xs, ys, tids_s, starts, ends = create_overlapping_windows(X, y, tids, 251, 50)
    assert Xs.shape == (n_trials * 6, 251, 15)
    assert len(ys) == n_trials * 6
    assert ends[0] - starts[0] == 251
    for i in range(n_trials):
        mask = tids_s == tids[i]
        assert np.all(ys[mask] == y[i])
        assert mask.sum() == 6
