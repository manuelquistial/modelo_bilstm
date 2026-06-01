#!/usr/bin/env python3
"""Train and evaluate all baseline models."""

from __future__ import annotations

import argparse

import pandas as pd

import scripts._bootstrap  # noqa: F401

from src.baselines.csp_svm import CSPSVMClassifier
from src.baselines.knn_baseline import KNNClassifier
from src.datasets.data_loader import prepare_subject_data
from src.evaluation.evaluator import evaluate_sklearn_model, save_confusion_figures, save_predictions, save_trial_predictions
from src.training.train_subject import metrics_row, train_subject
from src.training.train_all_subjects import list_subjects
from src.utils.config import load_config, project_root
from src.utils.io import append_or_create_metrics_csv
from src.utils.logging import setup_logger


def run_sklearn_baseline(subject_id: str, model_name: str, prepared: dict, model_cfg: dict, results_dir) -> dict:
    train, test = prepared["train"], prepared["test"]
    sfreq = prepared["sfreq"]
    if model_name == "csp_svm":
        clf = CSPSVMClassifier(model_cfg.get("csp_svm", {}))
    elif model_name in ("knn", "eff_knn"):
        clf = KNNClassifier(model_cfg.get("eff_knn", model_cfg.get("knn", {})), sfreq=sfreq)
    else:
        raise ValueError(model_name)
    clf.fit(train["X"], train["y"])
    ev = evaluate_sklearn_model(clf, test["X"], test["y"], test["trial_ids"])
    ev["y_true_seg"] = test["y"]
    save_predictions(subject_id, model_name, "test", test["y"], ev["y_pred_seg"], ev["probas_seg"], test["trial_ids"], None, results_dir)
    save_trial_predictions(subject_id, model_name, ev["unique_trial_ids"], ev["y_true_trial"], ev["y_pred_trial"], ev["probas_seg"], test["trial_ids"], results_dir)
    save_confusion_figures(ev, subject_id, model_name, results_dir, "segment")
    return metrics_row(subject_id, model_name, ev, len(prepared["train_trials_idx"]), len(prepared["test_trials_idx"]), len(train["y"]), len(test["y"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--quick-epochs", type=int, default=5, help="Quick training for DL models")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    dataset_cfg = load_config(root / "configs" / "dataset.yaml")
    model_cfg = load_config(root / "configs" / "model.yaml")
    training_cfg = load_config(root / "configs" / "training.yaml")
    processed_dir = root / training_cfg.get("processed_data_dir", "data/processed")
    results_dir = root / training_cfg.get("results_dir", "results")

    subjects = list_subjects(processed_dir) if args.all_subjects else [args.subject]
    if not subjects or subjects == [None]:
        raise ValueError("Specify --subject or --all-subjects")

    dl_models = ["eegnet", "convnet", "cnn_lstm", "proposed"]
    sklearn_models = ["csp_svm", "eff_knn"]
    all_rows = []

    for sid in subjects:
        if not sid:
            continue
        prepared = prepare_subject_data(sid, processed_dir, dataset_cfg)
        for m in sklearn_models:
            logger.info("%s — %s", sid, m)
            row = run_sklearn_baseline(sid, m, prepared, model_cfg, results_dir)
            all_rows.append(row)
            append_or_create_metrics_csv(pd.DataFrame([row]), results_dir / "metrics" / "per_subject_metrics.csv")
        for m in dl_models:
            logger.info("%s — %s (DL)", sid, m)
            result = train_subject(sid, m, quick_epochs=args.quick_epochs)
            all_rows.append(result["metrics"])

    summary = pd.DataFrame(all_rows)
    summary_path = results_dir / "metrics" / "all_models_summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        summary = pd.concat([old, summary], ignore_index=True).drop_duplicates(
            subset=["subject_id", "model_name"], keep="last"
        )
    summary.to_csv(summary_path, index=False)
    logger.info("Saved %s", summary_path)


if __name__ == "__main__":
    main()
