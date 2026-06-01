#!/usr/bin/env python3
"""Run PSO hyperparameter search for one subject."""

from __future__ import annotations

import argparse

import scripts._bootstrap  # noqa: F401

from src.optimization.pso import ParticleSwarmOptimizer, save_pso_results
from src.optimization.pso_objective import make_pso_fitness
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/pso.yaml")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    pso_cfg = load_config(root / args.config)
    training_cfg = load_config(root / "configs" / "training.yaml")
    training_cfg["fitness_epochs"] = pso_cfg.get("fitness_epochs", 30)
    model_cfg = load_config(root / "configs" / "model.yaml")
    dataset_cfg = load_config(root / "configs" / "dataset.yaml")
    processed_dir = root / training_cfg.get("processed_data_dir", "data/processed")

    fitness = make_pso_fitness(args.subject, processed_dir, dataset_cfg, model_cfg, training_cfg)
    pso = ParticleSwarmOptimizer(
        bounds=pso_cfg["bounds"],
        n_particles=pso_cfg.get("num_particles", 8),
        n_iterations=pso_cfg.get("num_iterations", 10),
        w=pso_cfg.get("w", 0.729),
        c1=pso_cfg.get("c1", 1.49445),
        c2=pso_cfg.get("c2", 1.49445),
    )
    best, history = pso.optimize(fitness)
    out_dir = root / pso_cfg.get("results_dir", "results/pso")
    save_pso_results(args.subject, best, history, out_dir)
    logger.info("PSO best params for %s: %s", args.subject, best)


if __name__ == "__main__":
    main()
