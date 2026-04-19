"""Common aggregate metrics used throughout UNCP evaluation."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def worst_group_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
) -> Tuple[float, Dict[int, float]]:
    per_group: Dict[int, float] = {}
    for g in torch.unique(groups).tolist():
        m = (groups == g)
        if m.any():
            per_group[int(g)] = float((preds[m] == labels[m]).float().mean().item())
    wga = min(per_group.values()) if per_group else 0.0
    return wga, per_group


def average_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return float((preds == labels).float().mean().item())


def delta_nsp(rank_before: Iterable[Tuple[str, float]],
              rank_after: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    b = dict(rank_before); a = dict(rank_after)
    return {nt: float(a.get(nt, 0.0) - b.get(nt, 0.0)) for nt in b}
