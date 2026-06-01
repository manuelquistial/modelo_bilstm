"""Confusion matrix plotting and saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
    normalize: bool = False,
) -> np.ndarray:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_plot, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            val = cm[i, j] if not normalize else cm_plot[i, j]
            text = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, text, ha="center", va="center")
    fig.colorbar(im, ax=ax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm
