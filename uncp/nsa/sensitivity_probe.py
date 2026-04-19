"""Core NSA diagnostic — probe a model with noise at many magnitudes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uncp.utils.io import load_pickle, save_pickle


DEFAULT_MAGNITUDES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class NoiseSensitivityProfile:
    """A full sweep of flip-rate curves per noise type × magnitude × group."""

    model_name: str
    domain: str
    results: Dict[str, Dict[float, Dict[str, Any]]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    method: str = "NSA"
    magnitudes: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def get_most_sensitive_noise(self) -> Tuple[str, float]:
        """Return (noise_type, max flip rate over all magnitudes)."""
        best = ("", -1.0)
        for nt, per_mag in self.results.items():
            for mag, stats in per_mag.items():
                if stats["flip_rate"] > best[1]:
                    best = (nt, float(stats["flip_rate"]))
        return best

    def get_vulnerability_ranking(self) -> List[Tuple[str, float]]:
        rank = [(nt, max(s["flip_rate"] for s in per_mag.values()))
                for nt, per_mag in self.results.items()]
        return sorted(rank, key=lambda t: t[1], reverse=True)

    def get_group_disparity(self) -> Dict[str, float]:
        """For each noise type, max (worst-group flip / best-group flip)."""
        out: Dict[str, float] = {}
        for nt, per_mag in self.results.items():
            ratios = []
            for mag, stats in per_mag.items():
                pg = stats.get("per_group_flip") or {}
                if len(pg) >= 2:
                    vals = list(pg.values())
                    lo = max(min(vals), 1e-6)
                    ratios.append(max(vals) / lo)
            out[nt] = max(ratios) if ratios else 1.0
        return out

    def get_magnitude_at_threshold(
        self, noise_type: str, threshold: float = 0.5, group: Optional[int] = None,
    ) -> Optional[float]:
        """Smallest magnitude where flip rate (for given group, or overall)
        first exceeds ``threshold``."""
        per_mag = self.results.get(noise_type, {})
        for mag in sorted(per_mag.keys()):
            stats = per_mag[mag]
            if group is None:
                if stats["flip_rate"] >= threshold:
                    return float(mag)
            else:
                pg = stats.get("per_group_flip") or {}
                if pg.get(group, 0.0) >= threshold:
                    return float(mag)
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "NoiseSensitivityProfile":
        return load_pickle(path)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for nt, per_mag in self.results.items():
            for mag, stats in per_mag.items():
                row = {"noise_type": nt, "magnitude": float(mag),
                       "flip_rate": float(stats["flip_rate"]),
                       "confidence_drop": float(stats.get("confidence_drop", 0.0)),
                       "num_samples": int(stats.get("num_samples", 0))}
                for g, v in (stats.get("per_group_flip") or {}).items():
                    row[f"group_{g}_flip"] = float(v)
                rows.append(row)
        return pd.DataFrame(rows)


class SensitivityProbe:
    """Measure per-sample prediction flips under noise perturbation."""

    def __init__(
        self,
        model: nn.Module,
        noise_generators: Dict[str, Any],
        magnitude_levels: Optional[List[float]] = None,
        num_samples: int = 500,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
        model_name: str = "model",
        domain: str = "vision",
    ) -> None:
        self.model = model.eval()
        self.noise_generators = noise_generators
        self.magnitude_levels = magnitude_levels or DEFAULT_MAGNITUDES
        self.num_samples = int(num_samples)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.domain = domain

    @torch.no_grad()
    def _collect_probe_samples(
        self, dataloader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels, groups = [], [], []
        total = 0
        for batch in dataloader:
            x = batch["image"]; y = batch["label"]
            g = batch.get("group_label", torch.zeros_like(y))
            take = min(self.num_samples - total, x.size(0))
            images.append(x[:take]); labels.append(y[:take]); groups.append(g[:take])
            total += take
            if total >= self.num_samples:
                break
        return (
            torch.cat(images).to(self.device),
            torch.cat(labels).to(self.device),
            torch.cat(groups).to(self.device),
        )

    @torch.no_grad()
    def probe(self, dataloader: DataLoader) -> NoiseSensitivityProfile:
        x, y, g = self._collect_probe_samples(dataloader)
        n = x.size(0)
        clean_logits_chunks, clean_preds_chunks, clean_conf_chunks = [], [], []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            out = self.model(x[start:end])
            probs = F.softmax(out, dim=1)
            clean_logits_chunks.append(out)
            clean_preds_chunks.append(out.argmax(dim=1))
            clean_conf_chunks.append(probs.max(dim=1).values)
        clean_preds = torch.cat(clean_preds_chunks)
        clean_conf = torch.cat(clean_conf_chunks)

        profile = NoiseSensitivityProfile(
            model_name=self.model_name, domain=self.domain,
            magnitudes=list(self.magnitude_levels),
        )

        groups_unique = torch.unique(g).tolist()

        for nt, gen in self.noise_generators.items():
            profile.results[nt] = {}
            for mag in self.magnitude_levels:
                noisy_preds_chunks, noisy_conf_chunks = [], []
                for start in range(0, n, self.batch_size):
                    end = start + self.batch_size
                    x_noisy = gen.apply(x[start:end], float(mag))
                    out = self.model(x_noisy)
                    probs = F.softmax(out, dim=1)
                    noisy_preds_chunks.append(out.argmax(dim=1))
                    noisy_conf_chunks.append(probs.max(dim=1).values)
                noisy_preds = torch.cat(noisy_preds_chunks)
                noisy_conf = torch.cat(noisy_conf_chunks)

                flipped = (noisy_preds != clean_preds)
                flip_rate = float(flipped.float().mean().item())

                per_class: Dict[int, float] = {}
                for cls in torch.unique(y).tolist():
                    m = (y == cls)
                    if m.any():
                        per_class[int(cls)] = float(flipped[m].float().mean().item())

                per_group: Dict[int, float] = {}
                for gr in groups_unique:
                    m = (g == gr)
                    if m.any():
                        per_group[int(gr)] = float(flipped[m].float().mean().item())

                conf_drop = float((clean_conf - noisy_conf).mean().item())

                profile.results[nt][float(mag)] = {
                    "flip_rate": flip_rate,
                    "per_class_flip": per_class,
                    "per_group_flip": per_group,
                    "confidence_drop": conf_drop,
                    "num_samples": int(n),
                }

        return profile


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    print(f"probing: {args.checkpoint}")
