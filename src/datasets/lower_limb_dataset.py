"""PyTorch dataset for lower-limb MI segments."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class ChannelStandardizer:
    """Per-channel z-score normalization fitted on training data."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        """Fit on (n, time, channels)."""
        self.mean_ = X.mean(axis=(0, 1), keepdims=False)
        self.std_ = X.std(axis=(0, 1), keepdims=False) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (n, time, channels)."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def fit_channel_standardizer(X_train: np.ndarray) -> ChannelStandardizer:
    return ChannelStandardizer().fit(X_train)


def transform_channel_standardizer(
    X: np.ndarray, standardizer: ChannelStandardizer
) -> np.ndarray:
    return standardizer.transform(X)


class LowerLimbMIDataset(Dataset):
    """Dataset yielding (time, channels) segments and class labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)
