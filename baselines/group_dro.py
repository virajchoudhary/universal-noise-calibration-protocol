"""Group Distributionally Robust Optimization (Sagawa et al. 2020).

Requires group labels during training — a strict advantage over UNCP.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .erm import ERMTrainer


class GroupDROTrainer(ERMTrainer):
    """Minimize worst-group weighted loss with adaptive group weights."""

    def __init__(self, *args, eta: float = 0.01, num_groups: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eta = float(eta)
        self.num_groups = int(num_groups)
        self.group_weights = torch.full((num_groups,), 1.0 / num_groups,
                                        device=self.device)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            g = batch["group_label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(x)
            per_sample = F.cross_entropy(logits, y, reduction="none")
            # Per-group mean loss
            group_losses = torch.zeros(self.num_groups, device=self.device)
            for gr in range(self.num_groups):
                mask = g == gr
                if mask.any():
                    group_losses[gr] = per_sample[mask].mean()
            # Update group weights multiplicatively, then normalize
            with torch.no_grad():
                self.group_weights = self.group_weights * torch.exp(self.eta * group_losses.detach())
                self.group_weights = self.group_weights / self.group_weights.sum()
            loss = (self.group_weights * group_losses).sum()
            loss.backward()
            self.optimizer.step()
            loss_sum += float(per_sample.mean().item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)
        self.scheduler.step()
        return {"train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1)}
