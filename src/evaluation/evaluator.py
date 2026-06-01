"""Model evaluation (segment and trial level)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.confusion import plot_confusion_matrix
from src.evaluation.metrics import aggregate_trial_predictions, compute_metrics


@torch.no_grad()
def predict_proba_torch(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return class probabilities."""
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probas = []
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)


def evaluate_sklearn_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    trial_ids: np.ndarray,
) -> dict[str, Any]:
    """Evaluate classical model at segment and trial level."""
    probas = model.predict_proba(X_test)
    preds = np.argmax(probas, axis=1)
    seg_metrics = compute_metrics(y_test, preds, paper_sensitivity=True)
    u_trials, y_true_t, y_pred_t = aggregate_trial_predictions(trial_ids, y_test, probas)
    trial_metrics = compute_metrics(y_true_t, y_pred_t, paper_sensitivity=True)
    return {
        "segment": seg_metrics,
        "trial": trial_metrics,
        "y_pred_seg": preds,
        "probas_seg": probas,
        "trial_ids": trial_ids,
        "y_true_trial": y_true_t,
        "y_pred_trial": y_pred_t,
        "unique_trial_ids": u_trials,
    }


def evaluate_torch_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    trial_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate PyTorch model."""
    probas = predict_proba_torch(model, X_test, device, batch_size)
    preds = np.argmax(probas, axis=1)
    seg_metrics = compute_metrics(y_test, preds, paper_sensitivity=True)
    u_trials, y_true_t, y_pred_t = aggregate_trial_predictions(trial_ids, y_test, probas)
    trial_metrics = compute_metrics(y_true_t, y_pred_t, paper_sensitivity=True)
    return {
        "segment": seg_metrics,
        "trial": trial_metrics,
        "y_pred_seg": preds,
        "probas_seg": probas,
        "trial_ids": trial_ids,
        "y_true_trial": y_true_t,
        "y_pred_trial": y_pred_t,
        "unique_trial_ids": u_trials,
    }


def save_predictions(
    subject_id: str,
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray,
    trial_ids: np.ndarray,
    segment_indices: np.ndarray | None,
    results_dir: Path,
) -> None:
    """Save segment-level predictions CSV."""
    n = len(y_true)
    df = pd.DataFrame({
        "subject_id": subject_id,
        "trial_id": trial_ids,
        "segment_id": segment_indices if segment_indices is not None else np.arange(n),
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_left": probas[:, 0],
        "prob_right": probas[:, 1],
        "split": split,
    })
    path = results_dir / "metrics" / f"predictions_subject_{subject_id}_model_{model_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_trial_predictions(
    subject_id: str,
    model_name: str,
    unique_trial_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas_seg: np.ndarray,
    trial_ids_seg: np.ndarray,
    results_dir: Path,
) -> None:
    """Save trial-level aggregated predictions."""
    rows = []
    for tid in unique_trial_ids:
        mask = trial_ids_seg == tid
        mean_p = probas_seg[mask].mean(axis=0)
        rows.append({
            "subject_id": subject_id,
            "trial_id": int(tid),
            "true_label": int(y_true[list(unique_trial_ids).index(tid)] if tid in unique_trial_ids else y_true[0]),
            "pred_label": int(np.argmax(mean_p)),
            "prob_left_mean": float(mean_p[0]),
            "prob_right_mean": float(mean_p[1]),
            "n_segments": int(mask.sum()),
        })
    # Rebuild correctly
    rows = []
    for i, tid in enumerate(unique_trial_ids):
        mask = trial_ids_seg == tid
        mean_p = probas_seg[mask].mean(axis=0)
        rows.append({
            "subject_id": subject_id,
            "trial_id": int(tid),
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
            "prob_left_mean": float(mean_p[0]),
            "prob_right_mean": float(mean_p[1]),
            "n_segments": int(mask.sum()),
        })
    df = pd.DataFrame(rows)
    path = results_dir / "metrics" / f"trial_predictions_subject_{subject_id}_model_{model_name}.csv"
    df.to_csv(path, index=False)


def save_confusion_figures(
    eval_result: dict[str, Any],
    subject_id: str,
    model_name: str,
    results_dir: Path,
    level: str = "segment",
) -> None:
    """Save confusion matrix figure."""
    if level == "segment":
        y_true = eval_result.get("y_true_seg")
        if y_true is None:
            return
        y_pred = eval_result["y_pred_seg"]
    else:
        y_true = eval_result["y_true_trial"]
        y_pred = eval_result["y_pred_trial"]
    if y_true is None:
        return
    path = results_dir / "confusion_matrices" / f"confusion_matrix_subject_{subject_id}_{model_name}_{level}.png"
    plot_confusion_matrix(
        y_true, y_pred,
        ["Left MI", "Right MI"],
        f"{model_name} {subject_id} ({level})",
        path,
        normalize=True,
    )
