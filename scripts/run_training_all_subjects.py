#!/usr/bin/env python3
"""Train all processed subjects."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.training.train_all_subjects import train_all_subjects
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="proposed")
    parser.add_argument("--quick-epochs", type=int, default=None)
    args = parser.parse_args()
    setup_logger()
    train_all_subjects(model_name=args.model, quick_epochs=args.quick_epochs)


if __name__ == "__main__":
    main()
