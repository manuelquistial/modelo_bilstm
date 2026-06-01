"""CSP + SVM with grid search — Sun et al. §2.6 [38]."""

from __future__ import annotations

from typing import Any

import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def segments_to_mne_format(X: np.ndarray) -> np.ndarray:
    return np.transpose(X, (0, 2, 1))


class CSPSVMClassifier:
    """CSP + SVM with optional grid search (paper §2.6)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        n_comp = config.get("n_components", 4)
        base_pipeline = Pipeline([
            ("csp", CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True)),
        ])
        if config.get("grid_search", True):
            param_grid = config.get("param_grid", {
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", "auto", 0.01, 0.1],
                "svm__kernel": [config.get("svm_kernel", "rbf")],
            })
            cv = StratifiedKFold(
                n_splits=config.get("cv_folds", 3),
                shuffle=True,
                random_state=config.get("seed", 42),
            )
            self.model = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=config.get("n_jobs", -1),
            )
        else:
            self.model = base_pipeline
            self.model.set_params(svm__C=config.get("svm_C", 1.0),
                                  svm__gamma=config.get("svm_gamma", "scale"),
                                  svm__kernel=config.get("svm_kernel", "rbf"))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CSPSVMClassifier":
        self.model.fit(segments_to_mne_format(X_train), y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(segments_to_mne_format(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(segments_to_mne_format(X))
