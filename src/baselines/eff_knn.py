"""Eff-KNN baseline — Escobar et al. 2023 [39], energy-efficient KNN for EEG."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.sklearn_utils import extract_bandpower_features


class EffKNNClassifier:
    """
    Energy-efficient KNN for EEG (Sun et al. cite [39]).

    Uses bandpower + log-variance features (same signal info as typical EEG KNN pipelines)
    with ball-tree search and distance weighting for efficient neighbor queries.
    """

    def __init__(self, config: dict[str, Any] | None = None, sfreq: float = 125.0) -> None:
        config = config or {}
        self.sfreq = sfreq
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=config.get("n_neighbors", 5),
                weights=config.get("weights", "distance"),
                algorithm=config.get("algorithm", "ball_tree"),
                metric=config.get("metric", "minkowski"),
                p=config.get("p", 2),
                n_jobs=config.get("n_jobs", -1),
            )),
        ])

    def _features(self, X: np.ndarray) -> np.ndarray:
        return extract_bandpower_features(X, self.sfreq)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EffKNNClassifier":
        self.pipeline.fit(self._features(X_train), y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(self._features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(self._features(X))
