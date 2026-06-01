#!/usr/bin/env python3
"""ERD/ERS analysis."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.analysis.erd_ers import run_erd_subject
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--raw-dir", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    analysis_cfg = load_config(root / "configs" / "analysis.yaml")
    prep_cfg = load_config(root / "configs" / "preprocessing.yaml")
    raw_dir = root / (args.raw_dir or analysis_cfg.get("raw_data_dir", "data/sample"))
    out = root / analysis_cfg.get("results_erd_dir", "results/erd")
    run_erd_subject(args.subject, raw_dir, out, analysis_cfg, prep_cfg)
    logger.info("ERD analysis done for %s", args.subject)


if __name__ == "__main__":
    main()
