"""Train single subject."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.data_loader import make_dataloaders, prepare_subject_data
from src.evaluation.evaluator import (
    save_confusion_figures,
    save_predictions,
    save_trial_predictions,
)
from src.models.model_factory import (
    build_model,
    build_model_with_params,
    sync_model_input_channels,
)
from src.training.trainer import Trainer
from src.utils.config import load_config, project_root
from src.utils.device import get_device
from src.utils.io import append_or_create_metrics_csv, ensure_dir
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def metrics_row(
    subject_id: str,
    model_name: str,
    eval_result: dict[str, Any],
    n_train_trials: int,
    n_test_trials: int,
    n_train_seg: int,
    n_test_seg: int,
) -> dict[str, Any]:
    seg = eval_result["segment"]
    trial = eval_result["trial"]
    return {
        "subject_id": subject_id,
        "model_name": model_name,
        "accuracy_segment": seg["accuracy"],
        "accuracy_trial": trial["accuracy"],
        "kappa_segment": seg["kappa"],
        "kappa_trial": trial["kappa"],
        "sensitivity_macro_segment": seg.get("sensitivity_macro", seg.get("sensitivity_paper")),
        "sensitivity_macro_trial": trial.get("sensitivity_paper", trial.get("sensitivity_macro")),
        "sensitivity_paper_trial": trial.get("sensitivity_paper", trial.get("sensitivity_macro")),
        "sensitivity_left": trial["sensitivity_left"],
        "sensitivity_right": trial["sensitivity_right"],
        "n_train_trials": n_train_trials,
        "n_test_trials": n_test_trials,
        "n_train_segments": n_train_seg,
        "n_test_segments": n_test_seg,
    }


def train_subject(
    subject_id: str,
    model_name: str = "proposed",
    training_cfg: dict[str, Any] | None = None,
    model_cfg: dict[str, Any] | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    pso_params: dict[str, Any] | None = None,
    quick_epochs: int | None = None,
) -> dict[str, Any]:
    """Full training pipeline for one subject."""
    root = project_root()
    training_cfg = training_cfg or load_config(root / "configs" / "training.yaml")
    model_cfg = model_cfg or load_config(root / "configs" / "model.yaml")
    dataset_cfg = dataset_cfg or load_config(root / "configs" / "dataset.yaml")

    set_seed(training_cfg.get("seed", 42))
    processed_dir = root / training_cfg.get("processed_data_dir", "data/processed")
    results_dir = root / training_cfg.get("results_dir", "results")
    ensure_dir(results_dir)

    prepared = prepare_subject_data(
        subject_id,
        processed_dir,
        dataset_cfg,
        use_internal_val=training_cfg.get("use_internal_val", False),
    )

    n_channels = int(prepared["test_trials"]["X"].shape[-1])
    model_cfg = sync_model_input_channels(model_cfg, n_channels)

    loaders = make_dataloaders(
        prepared,
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 0),
    )
    device = get_device(training_cfg.get("device", "auto"))

    if model_name == "proposed" and pso_params:
        model = build_model_with_params("proposed", model_cfg, pso_params)
    else:
        model = build_model(model_name, model_cfg)

    epochs = quick_epochs or training_cfg.get("epochs", 400)
    trainer = Trainer(model, training_cfg, device, results_dir)
    trainer.fit(
        loaders["train"],
        epochs=epochs,
        val_loader=loaders.get("val"),
        subject_id=subject_id,
        model_name=model_name,
    )

    test = prepared["test"]
    eval_result = trainer.evaluate_subject(
        test["X"], test["y"], test["trial_ids"],
        batch_size=training_cfg.get("batch_size", 32),
    )
    eval_result["y_true_seg"] = test["y"]

    save_predictions(
        subject_id, model_name, "test",
        test["y"], eval_result["y_pred_seg"], eval_result["probas_seg"],
        test["trial_ids"], np.arange(len(test["y"])), results_dir,
    )
    save_trial_predictions(
        subject_id, model_name,
        eval_result["unique_trial_ids"],
        eval_result["y_true_trial"],
        eval_result["y_pred_trial"],
        eval_result["probas_seg"],
        test["trial_ids"],
        results_dir,
    )
    save_confusion_figures(eval_result, subject_id, model_name, results_dir, "segment")
    save_confusion_figures(eval_result, subject_id, model_name, results_dir, "trial")

    row = metrics_row(
        subject_id, model_name, eval_result,
        len(prepared["train_trials_idx"]),
        len(prepared["test_trials_idx"]),
        len(prepared["train"]["y"]),
        len(test["y"]),
    )
    append_or_create_metrics_csv(
        pd.DataFrame([row]),
        results_dir / "metrics" / "per_subject_metrics.csv",
    )
    logger.info(
        "%s %s — segment acc=%.3f trial acc=%.3f",
        subject_id, model_name, row["accuracy_segment"], row["accuracy_trial"],
    )
    return {"metrics": row, "eval": eval_result, "model": trainer.model}
