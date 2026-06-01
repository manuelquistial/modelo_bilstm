#!/usr/bin/env python3
"""Run EEG preprocessing for all subjects in input directory."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.preprocessing.preprocessing_pipeline import preprocess_subject
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml")
    parser.add_argument("--subject", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    prep_cfg = load_config(root / args.config)
    dataset_cfg = load_config(root / "configs" / "dataset.yaml")
    channel_names = dataset_cfg.get("channel_names")
    input_dir = root / args.input
    output_dir = root / args.output

    subjects = [args.subject] if args.subject else sorted(
        p.name for p in input_dir.iterdir() if p.is_dir()
    )
    for sid in subjects:
        logger.info("Preprocessing %s", sid)
        preprocess_subject(sid, input_dir, output_dir, prep_cfg, channel_names)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
