#!/usr/bin/env python3
"""Download and import BNCI2014_001 (MOABB) into data/processed/."""

from __future__ import annotations

import argparse
import json
import sys

import scripts._bootstrap  # noqa: F401

from src.datasets.bnci2014_001_loader import import_bnci2014_001
from src.utils.config import project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import BNCI2014_001 motor imagery (2-class L/R hand) via MOABB."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for per-subject trials.npz",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated MOABB subject ids (e.g. 1,2,3). Default: all 9.",
    )
    parser.add_argument("--n-classes", type=int, default=2, choices=[2, 4])
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--resample", type=float, default=125.0)
    parser.add_argument("--expected-samples", type=int, default=501)
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    out = root / args.output

    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",")]

    try:
        summary = import_bnci2014_001(
            out,
            subjects=subjects,
            n_classes=args.n_classes,
            tmin=args.tmin,
            tmax=args.tmax,
            resample=args.resample,
            expected_samples=args.expected_samples,
        )
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Import complete. Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
