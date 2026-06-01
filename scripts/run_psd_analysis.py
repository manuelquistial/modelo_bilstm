#!/usr/bin/env python3
"""PSD and topomap analysis."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.analysis.psd_analysis import run_psd_subject
from src.training.train_all_subjects import list_subjects
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--all-subjects", action="store_true")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    cfg = load_config(root / "configs" / "analysis.yaml")
    processed = root / cfg.get("processed_data_dir", "data/processed")
    out = root / cfg.get("results_psd_dir", "results/psd")

    subjects = [args.subject] if args.subject else list_subjects(processed)
    for sid in subjects:
        run_psd_subject(sid, processed, out, cfg)
    logger.info("PSD analysis done.")


if __name__ == "__main__":
    main()
