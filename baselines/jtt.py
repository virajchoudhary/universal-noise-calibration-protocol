"""Just Train Twice — Liu et al. 2021. No group labels needed."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from .erm import ERMTrainer


class JTTTrainer(ERMTrainer):
    """Two-phase: identify high-loss samples via short ERM, then upsample them."""

    def __init__(
        self, *args,
        num_epochs_identification: int = 2,
        upsample_factor: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_epochs_identification = int(num_epochs_identification)
        self.upsample_factor = int(upsample_factor)

    def _identify_hard_samples(self) -> torch.Tensor:
        """Train a short ERM and return per-sample loss after convergence."""
        id_model = copy.deepcopy(self.model)
        opt = torch.optim.AdamW(id_model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        for _ in range(self.num_epochs_identification):
            id_model.train()
            for batch in self.train_loader:
                x = batch["image"].to(self.device); y = batch["label"].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(id_model(x), y)
                loss.backward(); opt.step()
        id_model.eval()
        per_sample_loss = []
        with torch.no_grad():
            for batch in self.train_loader:
                x = batch["image"].to(self.device); y = batch["label"].to(self.device)
                losses = F.cross_entropy(id_model(x), y, reduction="none")
                per_sample_loss.append(losses.cpu())
        return torch.cat(per_sample_loss)

    def run(self) -> Dict[str, Any]:
        print("[jtt] phase 1: identification")
        losses = self._identify_hard_samples()
        threshold = losses.median()
        high_loss_mask = losses > threshold
        weights = torch.ones_like(losses)
        weights[high_loss_mask] = float(self.upsample_factor)
        sampler = WeightedRandomSampler(
            weights.tolist(), num_samples=len(weights), replacement=True,
        )
        self.train_loader = DataLoader(
            self.train_loader.dataset, batch_size=self.train_loader.batch_size,
            sampler=sampler, num_workers=getattr(self.train_loader, "num_workers", 0),
        )
        print(f"[jtt] phase 2: upsampling {int(high_loss_mask.sum().item())} "
              f"high-loss samples by {self.upsample_factor}×")
        return super().run()
