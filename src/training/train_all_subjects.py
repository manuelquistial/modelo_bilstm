"""Train all subjects."""

from __future__ import annotations

import logging
from pathlib import Path

from src.training.train_subject import train_subject
from src.utils.config import load_config, project_root

logger = logging.getLogger(__name__)


def list_subjects(processed_dir: Path) -> list[str]:
    return sorted([p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "trials.npz").exists()])


def train_all_subjects(model_name: str = "proposed", quick_epochs: int | None = None) -> None:
    root = project_root()
    training_cfg = load_config(root / "configs" / "training.yaml")
    processed_dir = root / training_cfg.get("processed_data_dir", "data/processed")
    subjects = list_subjects(processed_dir)
    if not subjects:
        raise FileNotFoundError(f"No processed subjects in {processed_dir}")
    for sid in subjects:
        logger.info("Training %s model=%s", sid, model_name)
        train_subject(sid, model_name=model_name, quick_epochs=quick_epochs)
