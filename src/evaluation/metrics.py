"""Classification metrics — Sun et al. 2026 eq. (10)–(13)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)


def sensitivity_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Sensitivity (recall) per class — eq. (13)."""
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return float(recalls[0]), float(recalls[1])


def sensitivity_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Paper Table 3: one sensitivity value per subject (mean of left/right class recall).
    """
    left, right = sensitivity_per_class(y_true, y_pred)
    return float((left + right) / 2.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paper_sensitivity: bool = True,
) -> dict[str, Any]:
    """Accuracy, Cohen κ, and sensitivity as in Sun et al. 2026."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    sens_left, sens_right = sensitivity_per_class(y_true, y_pred)
    sens_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    sens_paper = sensitivity_paper(y_true, y_pred) if paper_sensitivity else sens_macro
    return {
        "accuracy": acc,
        "kappa": kappa,
        "sensitivity_macro": sens_macro,
        "sensitivity_paper": sens_paper,
        "sensitivity_left": sens_left,
        "sensitivity_right": sens_right,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def aggregate_trial_predictions(
    trial_ids: np.ndarray,
    y_true: np.ndarray,
    probas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average segment softmax per trial (paper evaluation on 45 test trials)."""
    unique_trials = np.unique(trial_ids)
    y_true_trials = []
    y_pred_trials = []
    for tid in unique_trials:
        mask = trial_ids == tid
        mean_prob = probas[mask].mean(axis=0)
        y_pred_trials.append(int(np.argmax(mean_prob)))
        y_true_trials.append(int(y_true[mask][0]))
    return (
        unique_trials,
        np.array(y_true_trials),
        np.array(y_pred_trials),
    )
