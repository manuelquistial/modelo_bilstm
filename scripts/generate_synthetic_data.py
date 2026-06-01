#!/usr/bin/env python3
"""Generate synthetic EEG dataset for pipeline testing."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.datasets.synthetic_data import generate_all_subjects
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data")
    parser.add_argument("--output", type=str, default="data/sample")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger()
    cfg = load_config(project_root() / args.config)
    subjects = cfg.get("subjects")
    out = project_root() / args.output
    generate_all_subjects(out, subjects=subjects, seed=args.seed)
    logger.info("Done. Synthetic data at %s", out)


if __name__ == "__main__":
    main()
