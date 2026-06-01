"""Tests for BNCI2014_001 import helpers (no MOABB download)."""

from __future__ import annotations

import numpy as np

from src.datasets.bnci2014_001_loader import (
    BNCI2014_001_CHANNEL_NAMES,
    _pad_or_crop_time,
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
