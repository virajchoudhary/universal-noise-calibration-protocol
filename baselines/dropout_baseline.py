"""
Dropout baseline for UNCP comparison.

Implements standard training with varying dropout rates as a
noise-based baseline. This connects to Gal & Ghahramani (2016)
who proved dropout is equivalent to approximate Bayesian inference.

Key insight: Dropout is a form of uniform, uncalibrated noise
injection. Unlike UNCP, it does not target specific spurious
features and is not calibrated based on diagnostic analysis.
"""

import time
import copy
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from uncp.utils.seed import set_seed


def add_dropout_to_model(
    model: nn.Module,
    dropout_rate: float,
    position: str = "before_last",
) -> nn.Module:
    """Insert Dropout layers into an existing model.

    Parameters
    ----------
    model : nn.Module
        The base model (e.g., ResNet-18).
    dropout_rate : float
        Dropout probability (0.0 to 0.5).
    position : str
        Where to add dropout:
        - "before_last": Before the final fully-connected layer
        - "after_each_block": After each residual block
        - "input": As the first layer (input dropout)

    Returns
    -------
    nn.Module
        Model with dropout inserted.
    """
    if dropout_rate <= 0.0:
        return model

    model = copy.deepcopy(model)

    if position == "before_last":
        # Find the last Linear layer and insert dropout before it
        last_linear_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name

        if last_linear_name is not None:
            # Navigate to the parent and replace
            parts = last_linear_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            old_layer = getattr(parent, parts[-1])
            dropout_layer = nn.Dropout(p=dropout_rate)
            # Create a sequential: dropout then the original layer
            setattr(
                parent,
                parts[-1],
                nn.Sequential(dropout_layer, old_layer),
            )

    elif position == "input":
        # Prepend dropout to the model
        original_forward = model.forward

        def new_forward(x):
            if self.training:
                # Apply input dropout (drop entire pixels)
                mask = torch.bernoulli(
                    torch.ones_like(x) * (1 - dropout_rate)
                )
                x = x * mask / (1 - dropout_rate)
            return original_forward(x)

        model.forward = new_forward

    return model


class DropoutBaselineTrainer:
    """Standard training with dropout as noise injection baseline.

    Tests multiple dropout rates to evaluate whether uncalibrated
    stochastic regularization alone can mitigate spurious correlations.

    Parameters
    ----------
    model : nn.Module
        The base neural network (dropout will be added).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader.
    config : DictConfig
        Configuration object.
    dropout_rate : float
        Dropout probability (default 0.3).
    dropout_position : str
        Where to insert dropout ("before_last" or "after_each_block").
    device : str
        Device string.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        dropout_rate: float = 0.3,
        dropout_position: str = "before_last",
        device: str = "cpu",
    ):
        # Add dropout to model
        self.model = add_dropout_to_model(model, dropout_rate, dropout_position)
        self.model = self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.dropout_rate = dropout_rate
        self.dropout_position = dropout_position
        self.device = device

        # Training hyperparameters
        train_cfg = getattr(config, "training", None)
        self.epochs = getattr(train_cfg, "epochs", 10) if train_cfg else 10
        self.lr = getattr(train_cfg, "lr", 1e-3) if train_cfg else 1e-3
        self.weight_decay = getattr(train_cfg, "weight_decay", 1e-4) if train_cfg else 1e-4

        self.best_worst_group_acc = 0.0
        self.best_model_state = None
        self.results = {}

    def _unpack_batch(self, batch):
        """Support this repo's dict batches and tuple-style batches."""
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
            g = batch.get("group_label", torch.zeros_like(y))
            return x, y, g
        if len(batch) == 3:
            return batch
        if len(batch) == 4:
            x, y, g, _ = batch
            return x, y, g
        x, y = batch[0], batch[1]
        return x, y, torch.zeros_like(y)

    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        group_correct = {}
        group_total = {}

        with torch.no_grad():
            for batch in loader:
                x, y, g = self._unpack_batch(batch)

                x = x.to(self.device)
                y = y.to(self.device)
                g = g.to(self.device)

                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=1)

                total_loss += loss.item() * x.size(0)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

                for i in range(x.size(0)):
                    gi = g[i].item()
                    group_correct[gi] = group_correct.get(gi, 0) + int(preds[i] == y[i])
                    group_total[gi] = group_total.get(gi, 0) + 1

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        per_group_acc = {}
        for gi in sorted(group_total.keys()):
            per_group_acc[gi] = group_correct[gi] / max(group_total[gi], 1)

        worst_group_acc = min(per_group_acc.values()) if per_group_acc else 0.0

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "worst_group_accuracy": worst_group_acc,
            "per_group_accuracy": per_group_acc,
        }

    def run(self) -> Dict[str, Any]:
        """Execute dropout baseline training.

        Returns
        -------
        dict
            Results with all standard metrics.
        """
        set_seed(getattr(self.config, "seed", 42))
        start_time = time.time()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        print(f"[Dropout Baseline] rate={self.dropout_rate}, "
              f"position={self.dropout_position}")
        print(f"[Dropout Baseline] Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch in self.train_loader:
                x, y, _ = self._unpack_batch(batch)

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x.size(0)
                epoch_correct += (logits.argmax(dim=1) == y).sum().item()
                epoch_total += x.size(0)

            scheduler.step()

            train_acc = epoch_correct / max(epoch_total, 1)
            val_metrics = self._evaluate(self.val_loader)

            if val_metrics["worst_group_accuracy"] > self.best_worst_group_acc:
                self.best_worst_group_acc = val_metrics["worst_group_accuracy"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"train_acc={train_acc:.4f}, "
                    f"val_wga={val_metrics['worst_group_accuracy']:.4f}, "
                    f"val_avg={val_metrics['accuracy']:.4f}"
                )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        elapsed = time.time() - start_time

        test_metrics = self._evaluate(self.test_loader)
        val_metrics = self._evaluate(self.val_loader)

        results = {
            "method": f"dropout_{self.dropout_rate}",
            "dropout_rate": self.dropout_rate,
            "dropout_position": self.dropout_position,
            "average_accuracy": test_metrics["accuracy"],
            "worst_group_accuracy": test_metrics["worst_group_accuracy"],
            "per_group_accuracy": test_metrics["per_group_accuracy"],
            "val_average_accuracy": val_metrics["accuracy"],
            "val_worst_group_accuracy": val_metrics["worst_group_accuracy"],
            "srd": None,
            "srd_v2": None,
            "training_time_seconds": elapsed,
            "requires_group_labels": False,
            "num_hyperparameters": 2,  # dropout_rate, weight_decay
            "relative_training_cost": "1x",
            "epochs": self.epochs,
        }
        self.results = results

        print(f"\n[Dropout {self.dropout_rate}] Final Results:")
        print(f"  Average accuracy:  {results['average_accuracy']:.4f}")
        print(f"  Worst-group acc:   {results['worst_group_accuracy']:.4f}")
        print(f"  Per-group:         {results['per_group_accuracy']}")
        print(f"  Training time:     {elapsed:.1f}s")

        return results
