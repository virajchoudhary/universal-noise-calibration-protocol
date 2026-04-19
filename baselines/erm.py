"""Empirical Risk Minimization trainer — the primary baseline."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def _extract_batch(batch: Any) -> Dict[str, torch.Tensor]:
    """Normalize a batch into a dict with keys image/label/group_label."""
    if isinstance(batch, dict):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


class ERMTrainer:
    """Minimal, well-tested ERM trainer.

    Supports:
      - Cross-entropy loss with AdamW + cosine LR schedule
      - Per-epoch logging of train/val/test accuracy and worst-group accuracy
      - Optional mixed precision (CUDA only)
      - Optional W&B logging
      - Best-checkpoint saving based on worst-group accuracy
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        device: torch.device,
        shifted_test_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints",
        run_name: str = "erm",
        use_wandb: bool = False,
        transform_input: Optional[callable] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.shifted_test_loader = shifted_test_loader
        self.config = config
        self.device = device
        self.run_name = run_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.transform_input = transform_input

        tr = config.training
        self.epochs = int(tr.epochs)
        self.lr = float(tr.lr)
        self.weight_decay = float(tr.weight_decay)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs, 1),
        )

        self.use_amp = (
            bool(tr.get("mixed_precision", False)) and device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.use_wandb = use_wandb
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb_run = wandb.init(
                    project=str(config.wandb.project_name), name=run_name,
                    config=dict(config), reinit=True,
                )
            except Exception as exc:
                print(f"[warn] W&B disabled: {exc}")
                self.use_wandb = False

        self.best_wga = -1.0
        self.history: list[dict] = []

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.transform_input is not None:
            images = self.transform_input(images)
        return self.model(images)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self._forward(images)
                    loss = F.cross_entropy(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self._forward(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
            loss_sum += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += images.size(0)
        self.scheduler.step()
        return {"train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_groups = [], [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            groups = batch.get("group_label")
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(1)
            loss_sum += float(loss.item()) * images.size(0)
            correct += int((preds == labels).sum().item())
            total += images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            if groups is not None:
                all_groups.append(groups)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        group_accs: Dict[int, float] = {}
        worst_group = 0
        worst_acc = 1.0
        if all_groups:
            groups = torch.cat(all_groups)
            for g in torch.unique(groups).tolist():
                mask = groups == g
                if mask.any():
                    acc = float((preds[mask] == labels[mask]).float().mean().item())
                    group_accs[int(g)] = acc
                    if acc < worst_acc:
                        worst_acc = acc
                        worst_group = int(g)
        return {
            "loss": loss_sum / max(total, 1),
            "acc": correct / max(total, 1),
            "per_group_acc": group_accs,
            "worst_group_acc": worst_acc if group_accs else None,
            "worst_group": worst_group if group_accs else None,
        }

    def get_group_accuracies(
        self, preds: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
    ) -> Dict[str, Any]:
        group_accs: Dict[int, float] = {}
        for g in torch.unique(groups).tolist():
            mask = groups == g
            if mask.any():
                group_accs[int(g)] = float(
                    (preds[mask] == labels[mask]).float().mean().item()
                )
        worst = min(group_accs.values()) if group_accs else None
        return {"per_group_acc": group_accs, "worst_group_acc": worst}

    def run(self) -> Dict[str, Any]:
        start = time.time()
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
                torch.save({"model": self.model.state_dict(),
                            "epoch": epoch, "val": val},
                           self.checkpoint_dir / f"{self.run_name}_best.pt")
            if self.use_wandb and self._wandb_run is not None:
                self._wandb_run.log(log)
            print(f"epoch {epoch:02d} | tr_loss {tr['train_loss']:.4f} "
                  f"tr_acc {tr['train_acc']*100:.2f} | "
                  f"val_acc {val['acc']*100:.2f} wga {(val['worst_group_acc'] or 0)*100:.2f}")

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
