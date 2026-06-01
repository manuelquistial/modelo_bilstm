#!/usr/bin/env python3
"""Re-run evaluation from saved checkpoints."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.datasets.data_loader import prepare_subject_data
from src.evaluation.evaluator import evaluate_torch_model, save_confusion_figures
from src.models.model_factory import build_model
from src.training.checkpoints import load_checkpoint
from src.training.train_subject import metrics_row
from src.utils.config import load_config, project_root
from src.utils.device import get_device
from src.utils.io import append_or_create_metrics_csv
from src.utils.logging import setup_logger
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--model", type=str, default="proposed")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    training_cfg = load_config(root / "configs" / "training.yaml")
    model_cfg = load_config(root / "configs" / "model.yaml")
    dataset_cfg = load_config(root / "configs" / "dataset.yaml")
    processed_dir = root / training_cfg.get("processed_data_dir", "data/processed")
    results_dir = root / training_cfg.get("results_dir", "results")
    device = get_device(training_cfg.get("device", "auto"))

    prepared = prepare_subject_data(args.subject, processed_dir, dataset_cfg)
    model = build_model(args.model, model_cfg)
    ckpt = results_dir / "models" / f"{args.subject}_{args.model}_best.pt"
    if ckpt.exists():
        load_checkpoint(ckpt, model, device)
    else:
        logger.warning("No checkpoint at %s — using random weights", ckpt)

    test = prepared["test"]
    ev = evaluate_torch_model(model, test["X"], test["y"], test["trial_ids"], device)
    ev["y_true_seg"] = test["y"]
    row = metrics_row(args.subject, args.model, ev, len(prepared["train_trials_idx"]), len(prepared["test_trials_idx"]), len(prepared["train"]["y"]), len(test["y"]))
    append_or_create_metrics_csv(pd.DataFrame([row]), results_dir / "metrics" / "per_subject_metrics.csv")
    save_confusion_figures(ev, args.subject, args.model, results_dir, "trial")
    logger.info("Eval: segment acc=%.3f trial acc=%.3f", row["accuracy_segment"], row["accuracy_trial"])


if __name__ == "__main__":
    main()
