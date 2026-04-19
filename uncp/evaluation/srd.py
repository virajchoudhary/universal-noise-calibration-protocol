"""SRD — Spurious Robustness Degradation metric.

SRD quantifies how much of a model's drop under corruption is attributable
to reliance on spurious correlations rather than the corruption itself.
A spurious-correlated model degrades much more on the minority group;
a robust model degrades symmetrically across groups.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class SRDResult:
    srd: float                          # degradation gap (worst − best)
    srd_v2: float                       # variance across groups
    per_group_clean_acc: Dict[int, float] = field(default_factory=dict)
    per_group_corrupt_acc: Dict[int, float] = field(default_factory=dict)
    per_group_degradation: Dict[int, float] = field(default_factory=dict)
    worst_group: Optional[int] = None
    best_group: Optional[int] = None

    def describe(self) -> str:
        return (
            f"SRD = {self.srd:+.4f}   SRD_v2 (variance) = {self.srd_v2:+.4f}\n"
            f"  worst group {self.worst_group}: "
            f"clean {self.per_group_clean_acc.get(self.worst_group, 0)*100:.2f}% → "
            f"corrupt {self.per_group_corrupt_acc.get(self.worst_group, 0)*100:.2f}%\n"
            f"  best group  {self.best_group}: "
            f"clean {self.per_group_clean_acc.get(self.best_group, 0)*100:.2f}% → "
            f"corrupt {self.per_group_corrupt_acc.get(self.best_group, 0)*100:.2f}%"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class SRDCalculator:

    def __init__(
        self,
        model: nn.Module,
        clean_loader: DataLoader,
        corrupted_loader: DataLoader,
        device: torch.device | str = "cpu",
        group_key: str = "group_label",
    ) -> None:
        self.model = model.eval()
        self.clean_loader = clean_loader
        self.corrupted_loader = corrupted_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.group_key = group_key

    @torch.no_grad()
    def _per_group_acc(self, loader: DataLoader) -> Dict[int, float]:
        correct: Dict[int, int] = {}
        total: Dict[int, int] = {}
        for batch in loader:
            x = batch["image"].to(self.device); y = batch["label"].to(self.device)
            g = batch.get(self.group_key, torch.zeros_like(y))
            preds = self.model(x).argmax(1)
            for gr in torch.unique(g).tolist():
                m = (g == gr)
                if m.any():
                    correct[int(gr)] = correct.get(int(gr), 0) + int((preds[m.to(self.device)] == y[m.to(self.device)]).sum().item())
                    total[int(gr)] = total.get(int(gr), 0) + int(m.sum().item())
        return {g: correct[g] / max(total[g], 1) for g in correct}

    def compute(self) -> SRDResult:
        clean = self._per_group_acc(self.clean_loader)
        corrupt = self._per_group_acc(self.corrupted_loader)
        common = sorted(set(clean) & set(corrupt))
        deg = {g: clean[g] - corrupt[g] for g in common}
        if not deg:
            return SRDResult(srd=0.0, srd_v2=0.0)
        worst_g = max(deg, key=lambda g: deg[g])
        best_g = min(deg, key=lambda g: deg[g])
        srd = deg[worst_g] - deg[best_g]
        srd_v2 = float(np.var(list(deg.values())))
        return SRDResult(
            srd=float(srd), srd_v2=srd_v2,
            per_group_clean_acc=clean, per_group_corrupt_acc=corrupt,
            per_group_degradation=deg,
            worst_group=worst_g, best_group=best_g,
        )


def create_corrupted_test_set(dataset_name: str, **kwargs):
    """Return the standard corrupted / anti-correlated evaluation set.

    For Colored MNIST, the "corruption" is the anti-correlated split (ρ = 0).
    For CIFAR-10, apply standard CIFAR-10-C corruption at a chosen severity.
    """
    if dataset_name == "colored_mnist":
        from uncp.data.colored_mnist import ColoredMNIST
        return ColoredMNIST(correlation_strength=0.0, split="test", download=False,
                            seed=kwargs.get("seed", 42),
                            label_noise=kwargs.get("label_noise", 0.25))
    raise ValueError(f"corruption preset not implemented for {dataset_name}")
