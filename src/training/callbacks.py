"""Training callbacks."""

from __future__ import annotations


class EarlyStopping:
    """Track best validation accuracy."""

    def __init__(self, patience: int = 20) -> None:
        self.patience = patience
        self.best_score = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
