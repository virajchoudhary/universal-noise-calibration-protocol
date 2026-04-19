"""Mixup — Zhang et al. 2018. Non-targeted convex augmentation."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .erm import ERMTrainer


class MixupTrainer(ERMTrainer):
    def __init__(self, *args, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            x = batch["image"].to(self.device); y = batch["label"].to(self.device)
            lam = float(np.random.beta(self.alpha, self.alpha))
            idx = torch.randperm(x.size(0), device=self.device)
            x_mix = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(x_mix)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward(); self.optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y_a).sum().item())
            total += x.size(0)
        self.scheduler.step()
        return {"train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1)}
