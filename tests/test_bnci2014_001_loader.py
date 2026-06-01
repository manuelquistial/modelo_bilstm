"""Tests for BNCI2014_001 import helpers (no MOABB download)."""

from __future__ import annotations

import numpy as np

from src.datasets.bnci2014_001_loader import (
    BNCI2014_001_CHANNEL_NAMES,
    _pad_or_crop_time,
    encode_and_filter_labels,
)


def test_pad_or_crop_time_crop():
    X = np.ones((2, 600, 22), dtype=np.float32)
    out = _pad_or_crop_time(X, 501)
    assert out.shape == (2, 501, 22)


def test_pad_or_crop_time_pad():
    X = np.ones((2, 490, 22), dtype=np.float32)
    out = _pad_or_crop_time(X, 501)
    assert out.shape == (2, 501, 22)


def test_channel_list_length():
    assert len(BNCI2014_001_CHANNEL_NAMES) == 22


def test_encode_labels_strings_two_class():
    y = np.array(["left_hand", "right_hand", "tongue", "left_hand"])
    X = np.arange(4)[:, None, None]  # dummy
    encoded, keep, X_f, _ = encode_and_filter_labels(
        y, ["left_hand", "right_hand"], X=X
    )
    assert keep.sum() == 3
    assert len(encoded) == 3
    assert list(encoded) == [0, 1, 0]
    assert X_f.shape[0] == 3


def test_encode_labels_event_codes():
    y = np.array([1, 2, 3])
    encoded, keep, _, _ = encode_and_filter_labels(y, ["left_hand", "right_hand"])
    assert list(encoded) == [0, 1]
    assert keep.sum() == 2
