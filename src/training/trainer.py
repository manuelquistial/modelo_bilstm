"""PyTorch trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.evaluator import evaluate_torch_model
from src.training.checkpoints import save_checkpoint
from src.training.losses import get_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Train and evaluate deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device,
        results_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.results_dir = Path(results_dir)
        self.criterion = get_loss(config.get("loss", "cross_entropy"))
        lr = config.get("learning_rate", 0.01)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=config.get("weight_decay", 0.0),
        )
        self.lr_drop_epoch = config.get("lr_drop_epoch", 300)
        self.lr_after = config.get("learning_rate_after_drop", 0.001)

    def _maybe_adjust_lr(self, epoch: int) -> None:
        if epoch == self.lr_drop_epoch:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr_after
            logger.info("Learning rate dropped to %s at epoch %d", self.lr_after, epoch)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            clip = self.config.get("gradient_clip_norm")
            if clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
        return correct / max(total, 1)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int | None = None,
        val_loader: DataLoader | None = None,
        subject_id: str = "S01",
        model_name: str = "proposed",
    ) -> nn.Module:
        epochs = epochs or self.config.get("epochs", 400)
        best_acc = -1.0
        ckpt_path = self.results_dir / "models" / f"{subject_id}_{model_name}_best.pt"

        for epoch in range(1, epochs + 1):
            self._maybe_adjust_lr(epoch)
            loss = self.train_epoch(train_loader)
            acc = self.evaluate_loader(val_loader or train_loader)
            if epoch % 50 == 0 or epoch == 1:
                logger.info("Epoch %d/%d loss=%.4f acc=%.4f", epoch, epochs, loss, acc)
            if acc > best_acc and self.config.get("save_best", True):
                best_acc = acc
                save_checkpoint(ckpt_path, self.model, self.optimizer, epoch, {"acc": acc})

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
        return self.model

    def evaluate_subject(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        trial_ids: np.ndarray,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        return evaluate_torch_model(
            self.model, X_test, y_test, trial_ids, self.device, batch_size
        )
