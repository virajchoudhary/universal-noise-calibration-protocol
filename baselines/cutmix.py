"""CutMix — Yun et al. 2019."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .erm import ERMTrainer


class CutMixTrainer(ERMTrainer):
    def __init__(self, *args, beta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    @staticmethod
    def _rand_bbox(size, lam):
        _, _, h, w = size
        cut_rat = float(np.sqrt(1.0 - lam))
        cw = int(w * cut_rat); ch = int(h * cut_rat)
        cx = np.random.randint(w); cy = np.random.randint(h)
        x1 = int(np.clip(cx - cw // 2, 0, w)); x2 = int(np.clip(cx + cw // 2, 0, w))
        y1 = int(np.clip(cy - ch // 2, 0, h)); y2 = int(np.clip(cy + ch // 2, 0, h))
        return x1, y1, x2, y2

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            x = batch["image"].to(self.device); y = batch["label"].to(self.device)
            lam = float(np.random.beta(self.beta, self.beta))
            idx = torch.randperm(x.size(0), device=self.device)
            x_mix = x.clone()
            x1, y1, x2, y2 = self._rand_bbox(x.size(), lam)
            x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1)) / (x.size(-1) * x.size(-2))
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
