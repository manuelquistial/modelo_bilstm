#!/usr/bin/env python3
"""Train model for a single subject."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.training.train_subject import train_subject
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--model", type=str, default="proposed")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--quick-epochs", type=int, default=None)
    parser.add_argument("--pso-params", type=str, default=None, help="JSON path with PSO best params")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    training_cfg = load_config(root / args.config)
    if args.quick_epochs:
        training_cfg["epochs"] = args.quick_epochs

    pso_params = None
    if args.pso_params:
        import json
        with open(args.pso_params) as f:
            pso_params = json.load(f)

    train_subject(
        args.subject,
        model_name=args.model,
        training_cfg=training_cfg,
        pso_params=pso_params,
        quick_epochs=args.quick_epochs,
    )
    logger.info("Training finished for %s", args.subject)


if __name__ == "__main__":
    main()
