#!/usr/bin/env python3
"""Generate summary report and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts._bootstrap  # noqa: F401

from src.analysis.visualization import plot_accuracy_by_subject, plot_model_comparison
from src.evaluation.confusion import plot_confusion_matrix
from src.evaluation.report_tables import build_summary_table
from src.utils.config import project_root
from src.utils.constants import PAPER_REFERENCE_METRICS
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    _ = args

    logger = setup_logger()
    root = project_root()
    results = root / "results"
    metrics_path = results / "metrics" / "per_subject_metrics.csv"
    reports = results / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        logger.error("No metrics at %s — run training first", metrics_path)
        return

    df = pd.read_csv(metrics_path)
    summary = build_summary_table(results / "metrics")
    summary.to_csv(reports / "summary.csv", index=False)

    plot_accuracy_by_subject(df, "proposed", figures / "accuracy_by_subject_proposed.png")
    if not summary.empty:
        plot_model_comparison(summary, "accuracy", figures / "model_comparison_accuracy.png")
        plot_model_comparison(summary, "kappa", figures / "model_comparison_kappa.png")
        plot_model_comparison(summary, "sensitivity", figures / "model_comparison_sensitivity.png")

    # Global confusion for proposed (trial level)
    pred_files = list((results / "metrics").glob("trial_predictions_*_proposed.csv"))
    if pred_files:
        all_p = pd.concat([pd.read_csv(f) for f in pred_files])
        plot_confusion_matrix(
            all_p["true_label"].values,
            all_p["pred_label"].values,
            ["Left MI", "Right MI"],
            "Global proposed (trial)",
            figures / "confusion_matrix_global_proposed.png",
            normalize=True,
        )

    # PSO history plot if exists
    pso_hist = root / "results" / "pso" / "history_subject_S01.csv"
    if pso_hist.exists():
        h = pd.read_csv(pso_hist)
        fig, ax = plt.subplots()
        for p in h["particle"].unique():
            sub = h[h["particle"] == p]
            ax.plot(sub["iteration"], sub["fitness"], alpha=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness (val acc)")
        ax.set_title("PSO history S01")
        fig.savefig(figures / "pso_history_subject_S01.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    md_lines = [
        "# Sun et al. 2026 — Results Report\n",
        "## Average metrics by model\n",
        summary.to_markdown(index=False) if hasattr(summary, "to_markdown") else summary.to_string(),
        "\n## Paper reference (do not expect exact match on synthetic data)\n",
    ]
    for model, ref in PAPER_REFERENCE_METRICS.items():
        md_lines.append(f"- **{model}**: acc={ref['accuracy']:.3f}, kappa={ref['kappa']:.3f}, sens={ref['sensitivity']:.3f}\n")

    (reports / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Report saved to %s", reports)


if __name__ == "__main__":
    main()
