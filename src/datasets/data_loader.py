"""Data loading — paper protocol: augmentation only on training set (§2.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.lower_limb_dataset import (
    LowerLimbMIDataset,
    fit_channel_standardizer,
    transform_channel_standardizer,
)
from src.datasets.subject_split import split_trials_stratified, split_trials_with_internal_val
from src.preprocessing.augmentation import create_overlapping_windows
from src.utils.io import load_trials_npz


def load_subject_trials(processed_dir: Path, subject_id: str) -> dict[str, Any]:
    path = processed_dir / subject_id / "trials.npz"
    if not path.exists():
        raise FileNotFoundError(f"Processed trials not found: {path}")
    return load_trials_npz(path)


def prepare_subject_data(
    subject_id: str,
    processed_dir: Path,
    dataset_cfg: dict[str, Any],
    use_internal_val: bool = False,
) -> dict[str, Any]:
    """
    Paper protocol (Sun et al. §2.3–2.4):
    - Stratified 70/30 split on trials BEFORE any windowing.
    - Overlapping windows applied ONLY to training trials (data expansion).
    - Test trials kept at trial level; segments derived only for forward pass / aggregation.
    - No channel z-score unless channel_normalize=true (not in original MATLAB pipeline).
    """
    data = load_subject_trials(processed_dir, subject_id)
    X, y = data["X"], data["y"]
    trial_ids = data["trial_ids"]
    window_size = dataset_cfg.get("window_size", 251)
    step_size = dataset_cfg.get("step_size", 50)
    train_ratio = dataset_cfg.get("train_ratio", 0.70)
    seed = dataset_cfg.get("split_seed", 42)
    paper_mode = dataset_cfg.get("paper_mode", True)
    do_normalize = dataset_cfg.get("channel_normalize", False)

    if use_internal_val:
        train_idx, val_idx, test_idx = split_trials_with_internal_val(
            y, train_ratio, dataset_cfg.get("val_ratio_from_train", 0.15), seed
        )
    else:
        train_idx, test_idx = split_trials_stratified(y, trial_ids, train_ratio, seed)
        val_idx = np.array([], dtype=np.int64)

    def segment_trials(idxs: np.ndarray) -> dict[str, np.ndarray]:
        if len(idxs) == 0:
            return {
                "X": np.zeros((0, window_size, X.shape[2]), dtype=np.float32),
                "y": np.array([], dtype=np.int64),
                "trial_ids": np.array([], dtype=np.int64),
                "starts": np.array([], dtype=np.int64),
                "ends": np.array([], dtype=np.int64),
            }
        Xs, ys, tids, starts, ends = create_overlapping_windows(
            X[idxs], y[idxs], trial_ids[idxs], window_size, step_size
        )
        return {"X": Xs, "y": ys, "trial_ids": tids, "starts": starts, "ends": ends}

    # Training: overlapping windows (6× expansion per trial)
    train_seg = segment_trials(train_idx)
    val_seg = segment_trials(val_idx) if len(val_idx) else None

    # Test: trial-level arrays + segments for segment-wise inference
    test_trials = {
        "X": X[test_idx].astype(np.float32),
        "y": y[test_idx],
        "trial_ids": trial_ids[test_idx],
    }
    test_seg = segment_trials(test_idx) if not paper_mode else segment_trials(test_idx)

    standardizer = None
    if do_normalize:
        standardizer = fit_channel_standardizer(train_seg["X"])
        train_seg["X"] = transform_channel_standardizer(train_seg["X"], standardizer)
        test_seg["X"] = transform_channel_standardizer(test_seg["X"], standardizer)
        test_trials["X"] = transform_channel_standardizer(test_trials["X"], standardizer)
        if val_seg is not None and len(val_seg["X"]):
            val_seg["X"] = transform_channel_standardizer(val_seg["X"], standardizer)

    return {
        "subject_id": subject_id,
        "train_trials_idx": train_idx,
        "test_trials_idx": test_idx,
        "val_trials_idx": val_idx,
        "train": train_seg,
        "test": test_seg,
        "test_trials": test_trials,
        "val": val_seg,
        "standardizer": standardizer,
        "sfreq": data["sfreq"],
        "channel_names": data["channel_names"],
        "X_trials": X,
        "y_trials": y,
        "trial_ids_all": trial_ids,
        "paper_mode": paper_mode,
    }


def make_dataloaders(
    prepared: dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ("train", "val"):
        seg = prepared.get(split)
        if seg is None or len(seg.get("y", [])) == 0:
            continue
        ds = LowerLimbMIDataset(seg["X"], seg["y"])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers
        )
    # Test loader uses segments for batched inference (metrics at trial level separately)
    test_seg = prepared.get("test")
    if test_seg is not None and len(test_seg.get("y", [])):
        ds = LowerLimbMIDataset(test_seg["X"], test_seg["y"])
        loaders["test"] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loaders
