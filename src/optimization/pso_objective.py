"""PSO fitness function using validation accuracy."""

from __future__ import annotations

import logging
from typing import Any

from src.datasets.data_loader import make_dataloaders, prepare_subject_data
from src.datasets.subject_split import split_trials_with_internal_val
from src.models.model_factory import build_model_with_params
from src.preprocessing.augmentation import create_overlapping_windows
from src.datasets.lower_limb_dataset import fit_channel_standardizer, transform_channel_standardizer
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def make_pso_fitness(
    subject_id: str,
    processed_dir: Any,
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> Any:
    """Return fitness callable for PSO."""
    set_seed(training_cfg.get("seed", 42))
    data = __import__("src.utils.io", fromlist=["load_trials_npz"]).load_trials_npz(
        processed_dir / subject_id / "trials.npz"
    )
    X, y = data["X"], data["y"]
    trial_ids = data["trial_ids"]
    train_idx, val_idx, _ = split_trials_with_internal_val(
        y, dataset_cfg.get("train_ratio", 0.7),
        training_cfg.get("val_ratio_from_train", 0.15),
        dataset_cfg.get("split_seed", 42),
    )
    window_size = dataset_cfg.get("window_size", 251)
    step_size = dataset_cfg.get("step_size", 50)

    def segment(idxs):
        return create_overlapping_windows(
            X[idxs], y[idxs], trial_ids[idxs], window_size, step_size
        )

    X_tr, y_tr, _, _, _ = segment(train_idx)
    X_va, y_va, _, _, _ = segment(val_idx)
    std = fit_channel_standardizer(X_tr)
    X_tr = transform_channel_standardizer(X_tr, std)
    X_va = transform_channel_standardizer(X_va, std)

    prepared = {
        "train": {"X": X_tr, "y": y_tr},
        "val": {"X": X_va, "y": y_va},
    }
    loaders = make_dataloaders(prepared, training_cfg.get("batch_size", 32), 0)
    device = get_device(training_cfg.get("device", "auto"))
    fitness_epochs = training_cfg.get("fitness_epochs", 30)

    def fitness(params: dict[str, int]) -> float:
        model = build_model_with_params("proposed", model_cfg, params)
        cfg = dict(training_cfg)
        cfg["epochs"] = fitness_epochs
        cfg["save_best"] = False
        trainer = Trainer(model, cfg, device, training_cfg.get("results_dir", "results"))
        trainer.fit(loaders["train"], epochs=fitness_epochs, val_loader=loaders["val"])
        return trainer.evaluate_loader(loaders["val"])

    return fitness
