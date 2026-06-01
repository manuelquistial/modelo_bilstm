"""Aggregate metrics tables for reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_summary_table(metrics_dir: Path) -> pd.DataFrame:
    """Load per-subject metrics and compute mean per model."""
    per_subject_path = metrics_dir / "per_subject_metrics.csv"
    if not per_subject_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(per_subject_path)
    agg = df.groupby("model_name").agg({
        "accuracy_segment": "mean",
        "accuracy_trial": "mean",
        "kappa_segment": "mean",
        "kappa_trial": "mean",
        "sensitivity_macro_segment": "mean",
        "sensitivity_paper_trial": "mean",
        "sensitivity_macro_trial": "mean",
    }).reset_index()
    agg.rename(columns={
        "accuracy_segment": "avg_accuracy_segment",
        "accuracy_trial": "avg_accuracy_trial",
        "kappa_segment": "avg_kappa_segment",
        "kappa_trial": "avg_kappa_trial",
        "sensitivity_macro_segment": "avg_sensitivity_segment",
        "sensitivity_paper_trial": "avg_sensitivity_trial",
    }, inplace=True)
    return agg
