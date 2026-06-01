"""Stratified train/test split per subject (trial-level, before segmentation)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_trials_stratified(
    y: np.ndarray,
    trial_ids: np.ndarray,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified split on trials.

    Returns train_indices, test_indices (indices into trial arrays).
    """
    n = len(y)
    indices = np.arange(n)
    test_size = 1.0 - train_ratio
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(indices, y))
    return train_idx, test_idx


def split_trials_with_internal_val(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio_from_train: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into train, val, test with stratification."""
    train_idx, test_idx = split_trials_stratified(y, np.arange(len(y)), train_ratio, seed)
    y_train = y[train_idx]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_from_train, random_state=seed + 1
    )
    sub_train, val_idx_local = next(splitter.split(np.arange(len(y_train)), y_train))
    val_idx = train_idx[val_idx_local]
    train_idx_final = train_idx[sub_train]
    return train_idx_final, val_idx, test_idx


def verify_no_trial_leakage(
    train_trial_ids: np.ndarray,
    test_trial_ids: np.ndarray,
) -> bool:
    """Ensure no overlapping trial IDs between splits."""
    overlap = set(train_trial_ids.tolist()) & set(test_trial_ids.tolist())
    return len(overlap) == 0
