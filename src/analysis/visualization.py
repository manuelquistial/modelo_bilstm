"""Report visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_by_subject(df: pd.DataFrame, model_name: str, save_path: Path) -> None:
    sub = df[df["model_name"] == model_name]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sub["subject_id"], sub["accuracy_trial"])
    ax.set_ylabel("Trial accuracy")
    ax.set_title(f"Accuracy by subject — {model_name}")
    ax.set_ylim(0, 1)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, metric: str, save_path: Path) -> None:
    col = f"avg_{metric}_trial" if f"avg_{metric}_trial" in summary.columns else metric
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["model_name"], summary[col])
    ax.set_ylabel(metric)
    ax.set_title(f"Average {metric} by model")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
