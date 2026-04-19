"""
PGD Adversarial Training baseline for UNCP comparison.

Implements Projected Gradient Descent adversarial training
(Madry et al., 2018) as a robustness baseline.

This is a NON-targeted noise baseline — the perturbation is
adversarially optimized for worst-case input, NOT calibrated
for spurious correlation removal.
"""

import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from uncp.utils.seed import set_seed


class AdversarialTrainer:
    """PGD Adversarial Training baseline.

    Trains the model on adversarially perturbed inputs. The perturbation
    is computed via projected gradient descent on the input to maximize
    the classification loss within an L-infinity epsilon ball.

    Key difference from UNCP: noise is adversarially optimized for
    worst-case robustness, not calibrated to target spurious features.
    This often improves robustness to adversarial examples but does not
    specifically address spurious correlations.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    train_loader : DataLoader
        Training data loader (must return x, y, group_label).
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader (typically anti-correlated for spurious eval).
    config : DictConfig
        Configuration object with training hyperparameters.
    device : str
        Device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # PGD hyperparameters
        adv_cfg = getattr(config, "adversarial", None)
        self.pgd_eps = getattr(adv_cfg, "pgd_eps", 8.0 / 255.0) if adv_cfg else 8.0 / 255.0
        self.pgd_alpha = getattr(adv_cfg, "pgd_alpha", 2.0 / 255.0) if adv_cfg else 2.0 / 255.0
        self.pgd_steps = getattr(adv_cfg, "pgd_steps", 10) if adv_cfg else 10

        # Training hyperparameters
        train_cfg = getattr(config, "training", None)
        self.epochs = getattr(train_cfg, "epochs", 10) if train_cfg else 10
        self.lr = getattr(train_cfg, "lr", 1e-3) if train_cfg else 1e-3
        self.weight_decay = getattr(train_cfg, "weight_decay", 1e-4) if train_cfg else 1e-4
        self.batch_size = getattr(train_cfg, "batch_size", 128) if train_cfg else 128

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

    def _pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PGD adversarial perturbation.

        Parameters
        ----------
        x : Tensor
            Clean input batch, shape (B, C, H, W).
        y : Tensor
            True labels, shape (B,).

        Returns
        -------
        Tensor
            Adversarially perturbed input within eps L-inf ball.
        """
        x_adv = x.clone().detach()

        # Random start within epsilon ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.pgd_eps, self.pgd_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.pgd_steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]
            # Ascend the gradient (maximize loss)
            x_adv = x_adv.detach() + self.pgd_alpha * grad.sign()
            # Project back to epsilon ball around original x
            delta = torch.clamp(x_adv - x, -self.pgd_eps, self.pgd_eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        return x_adv

    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on a data loader.

        Returns dict with overall accuracy, per-group accuracy,
        worst-group accuracy, and loss.
        """
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
        """Execute PGD adversarial training and return results.

        Returns
        -------
        dict
            Results including average accuracy, worst-group accuracy,
            per-group accuracy, SRD (placeholder), training time,
            and method metadata.
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

        print(f"[Adversarial Training] PGD eps={self.pgd_eps:.4f}, "
              f"alpha={self.pgd_alpha:.4f}, steps={self.pgd_steps}")
        print(f"[Adversarial Training] Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch in self.train_loader:
                x, y, _ = self._unpack_batch(batch)

                x = x.to(self.device)
                y = y.to(self.device)

                # Generate adversarial examples
                x_adv = self._pgd_attack(x, y)

                # Train on adversarial examples
                logits = self.model(x_adv)
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

            # Check for best model
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

        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        elapsed = time.time() - start_time

        # Final evaluation
        test_metrics = self._evaluate(self.test_loader)
        val_metrics = self._evaluate(self.val_loader)

        results = {
            "method": "adversarial_training",
            "average_accuracy": test_metrics["accuracy"],
            "worst_group_accuracy": test_metrics["worst_group_accuracy"],
            "per_group_accuracy": test_metrics["per_group_accuracy"],
            "val_average_accuracy": val_metrics["accuracy"],
            "val_worst_group_accuracy": val_metrics["worst_group_accuracy"],
            "srd": None,  # Computed by comparison runner
            "srd_v2": None,
            "training_time_seconds": elapsed,
            "requires_group_labels": False,
            "num_hyperparameters": 3,  # eps, alpha, steps
            "relative_training_cost": f"{self.pgd_steps + 1}x",  # PGD steps + 1 forward
            "pgd_eps": self.pgd_eps,
            "pgd_alpha": self.pgd_alpha,
            "pgd_steps": self.pgd_steps,
            "epochs": self.epochs,
        }
        self.results = results

        print(f"\n[Adversarial Training] Final Results:")
        print(f"  Average accuracy:  {results['average_accuracy']:.4f}")
        print(f"  Worst-group acc:   {results['worst_group_accuracy']:.4f}")
        print(f"  Per-group:         {results['per_group_accuracy']}")
        print(f"  Training time:     {elapsed:.1f}s")

        return results
