"""Particle Swarm Optimization for CNN-SE-BiLSTM hyperparameters."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARAM_NAMES = [
    "conv1_filters",
    "conv2_filters",
    "conv1_kernel_time",
    "conv1_kernel_channels",
    "conv2_kernel_time",
    "conv2_kernel_channels",
]


class ParticleSwarmOptimizer:
    """PSO with velocity/position updates per paper coefficients."""

    def __init__(
        self,
        bounds: dict[str, list[int]],
        n_particles: int = 8,
        n_iterations: int = 10,
        w: float = 0.729,
        c1: float = 1.49445,
        c2: float = 1.49445,
        seed: int = 42,
    ) -> None:
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)
        self.dim = len(PARAM_NAMES)
        self.low = np.array([bounds[k][0] for k in PARAM_NAMES], dtype=float)
        self.high = np.array([bounds[k][1] for k in PARAM_NAMES], dtype=float)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.low, self.high)

    def _to_params(self, position: np.ndarray) -> dict[str, int]:
        rounded = np.round(position).astype(int)
        rounded = np.clip(rounded, self.low.astype(int), self.high.astype(int))
        return {name: int(rounded[i]) for i, name in enumerate(PARAM_NAMES)}

    def optimize(
        self,
        fitness_fn: Callable[[dict[str, int]], float],
    ) -> tuple[dict[str, int], pd.DataFrame]:
        """Run PSO and return best params + history."""
        positions = self.rng.uniform(self.low, self.high, size=(self.n_particles, self.dim))
        velocities = self.rng.uniform(-1, 1, size=(self.n_particles, self.dim))
        pbest_pos = positions.copy()
        pbest_scores = np.full(self.n_particles, -np.inf)
        gbest_pos = positions[0].copy()
        gbest_score = -np.inf
        history_rows: list[dict[str, Any]] = []

        for it in range(self.n_iterations):
            for p in range(self.n_particles):
                params = self._to_params(positions[p])
                score = fitness_fn(params)
                if score > pbest_scores[p]:
                    pbest_scores[p] = score
                    pbest_pos[p] = positions[p].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_pos = positions[p].copy()
                row = {"iteration": it, "particle": p, "fitness": score, **params}
                history_rows.append(row)
                logger.info("Iter %d particle %d fitness=%.4f params=%s", it, p, score, params)

            r1 = self.rng.random((self.n_particles, self.dim))
            r2 = self.rng.random((self.n_particles, self.dim))
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            positions = self._clip(positions + velocities)

        best_params = self._to_params(gbest_pos)
        return best_params, pd.DataFrame(history_rows)


def save_pso_results(
    subject_id: str,
    best_params: dict[str, int],
    history: pd.DataFrame,
    results_dir: Path,
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"best_params_subject_{subject_id}.json").open("w") as f:
        json.dump(best_params, f, indent=2)
    history.to_csv(out / f"history_subject_{subject_id}.csv", index=False)
