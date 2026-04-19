"""Group Distributionally Robust Optimization (Sagawa et al. 2020).

Requires group labels during training — a strict advantage over UNCP.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .erm import ERMTrainer


class GroupDROTrainer(ERMTrainer):
    """Minimize worst-group weighted loss with adaptive group weights."""

    def __init__(self, *args, eta: float = 0.05, num_groups: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eta = float(eta)
        self.num_groups = int(num_groups)
        self.group_weights = torch.full((num_groups,), 1.0 / num_groups,
                                        device=self.device)

        # Group DRO tends to overfit the worst group — force stronger regularization.
        min_weight_decay = 0.01
        if self.weight_decay < min_weight_decay:
            self.weight_decay = min_weight_decay
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.epochs, 1),
            )

        self.early_stop_patience = 3

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

    def run(self) -> Dict[str, Any]:
        start = time.time()
        epochs_without_improvement = 0
        best_state: Dict[str, torch.Tensor] | None = None
        for epoch in range(self.epochs):
            tr = self._train_epoch(epoch)
            val = self.evaluate(self.val_loader)
            log = {
                "epoch": epoch,
                **tr,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_worst_group_acc": val["worst_group_acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(log)
            wga = val["worst_group_acc"] or 0.0
            if wga > self.best_wga:
                self.best_wga = wga
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                torch.save({"model": self.model.state_dict(),
                            "epoch": epoch, "val": val},
                           self.checkpoint_dir / f"{self.run_name}_best.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if self.use_wandb and self._wandb_run is not None:
                self._wandb_run.log(log)
            print(f"epoch {epoch:02d} | tr_loss {tr['train_loss']:.4f} "
                  f"tr_acc {tr['train_acc']*100:.2f} | "
                  f"val_acc {val['acc']*100:.2f} wga {(val['worst_group_acc'] or 0)*100:.2f}")
            if epochs_without_improvement >= self.early_stop_patience:
                print(f"[GroupDRO] Early stopping at epoch {epoch}: "
                      f"WGA did not improve for {self.early_stop_patience} epochs "
                      f"(best={self.best_wga:.4f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        test = self.evaluate(self.test_loader)
        shifted = self.evaluate(self.shifted_test_loader) if self.shifted_test_loader else None
        elapsed = time.time() - start
        results = {
            "run_name": self.run_name,
            "epochs": self.epochs,
            "history": self.history,
            "best_val_worst_group_acc": self.best_wga,
            "test": test,
            "shifted_test": shifted,
            "elapsed_sec": elapsed,
        }
        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.finish()
        return results
