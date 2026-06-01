"""Test evaluation metrics."""

import numpy as np

from src.evaluation.metrics import compute_metrics


def test_perfect_accuracy():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["kappa"] == 1.0
    assert m["sensitivity_paper"] == 1.0


def test_chance_level():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = compute_metrics(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert -1.0 <= m["kappa"] <= 1.0
