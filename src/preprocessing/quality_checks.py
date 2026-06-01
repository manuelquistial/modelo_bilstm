"""Data quality checks."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_epoch_shape(
    X: np.ndarray,
    expected_samples: int = 501,
    expected_channels: int = 15,
) -> bool:
    """Validate trial tensor shape."""
    if X.ndim != 3:
        logger.error("Expected 3D array, got shape %s", X.shape)
        return False
    n, t, c = X.shape
    ok = t == expected_samples and c == expected_channels
    if not ok:
        logger.warning("Shape mismatch: got (%d, %d, %d), expected (*, %d, %d)", n, t, c, expected_samples, expected_channels)
    return ok


def check_labels_binary(y: np.ndarray) -> bool:
    """Ensure binary labels {0, 1}."""
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({0, 1}):
        logger.error("Labels must be 0 or 1, got %s", unique)
        return False
    return True
