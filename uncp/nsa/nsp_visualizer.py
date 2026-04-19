"""Publication-quality visualizations for Noise Sensitivity Profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sensitivity_probe import NoiseSensitivityProfile


_STYLE_APPLIED = False


def _apply_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    _STYLE_APPLIED = True


class NSPVisualizer:

    def plot_flip_curves(
        self, nsp: NoiseSensitivityProfile, save_path: Optional[str] = None,
        num_classes: int = 10,
    ) -> plt.Figure:
        _apply_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.cm.viridis(np.linspace(0, 1, max(1, len(nsp.results))))
        for (nt, per_mag), c in zip(nsp.results.items(), cmap):
            mags = sorted(per_mag.keys())
            flips = [per_mag[m]["flip_rate"] for m in mags]
            ax.plot(mags, flips, marker="o", label=nt, color=c, linewidth=1.3)
        ax.axhline((num_classes - 1) / num_classes, ls="--", c="gray", lw=0.8,
                   label="random-guess baseline")
        ax.set_xlabel("noise magnitude σ")
        ax.set_ylabel("flip rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Noise Sensitivity Profile — {nsp.model_name}")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_group_disparity(
        self, nsp: NoiseSensitivityProfile, save_path: Optional[str] = None,
    ) -> plt.Figure:
        _apply_style()
        noise_types = list(nsp.results.keys())
        any_mag = next(iter(next(iter(nsp.results.values())).values()))
        groups = sorted(any_mag["per_group_flip"].keys())

        per_group_max: dict[int, list[float]] = {g: [] for g in groups}
        for nt in noise_types:
            per_mag = nsp.results[nt]
            for g in groups:
                per_group_max[g].append(max(m["per_group_flip"].get(g, 0.0)
                                             for m in per_mag.values()))

        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / max(1, len(groups))
        x = np.arange(len(noise_types))
        for i, g in enumerate(groups):
            ax.bar(x + i * width, per_group_max[g], width,
                   label=f"group {g}" + (" (minority)" if g == 1 else ""))
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(noise_types, rotation=20, ha="right")
        ax.set_ylabel("max flip rate across σ")
        ax.set_title("Per-group vulnerability under each noise type")
        ax.legend(frameon=False)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_before_after(
        self, nsp_before: NoiseSensitivityProfile,
        nsp_after: NoiseSensitivityProfile, save_path: Optional[str] = None,
    ) -> plt.Figure:
        _apply_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        noise_types = sorted(set(nsp_before.results) & set(nsp_after.results))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(1, len(noise_types))))
        for nt, c in zip(noise_types, cmap):
            mags_b = sorted(nsp_before.results[nt].keys())
            mags_a = sorted(nsp_after.results[nt].keys())
            flips_b = [nsp_before.results[nt][m]["flip_rate"] for m in mags_b]
            flips_a = [nsp_after.results[nt][m]["flip_rate"] for m in mags_a]
            ax.plot(mags_b, flips_b, "-", color=c, lw=1.2, label=f"{nt} (ERM)")
            ax.plot(mags_a, flips_a, "--", color=c, lw=1.2, label=f"{nt} (UNCP)")
        ax.set_xlabel("noise magnitude σ")
        ax.set_ylabel("flip rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("NSP — before vs. after UNCP treatment")
        ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def create_diagnostic_report(
        self, nsp: NoiseSensitivityProfile, output_dir: str | Path,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_flip_curves(nsp, save_path=output_dir / "nsp_flip_curves.pdf")
        plt.close()
        self.plot_group_disparity(nsp, save_path=output_dir / "nsp_group_disparity.pdf")
        plt.close()
        nsp.to_dataframe().to_csv(output_dir / "nsp_table.csv", index=False)

        ranking = nsp.get_vulnerability_ranking()
        disparity = nsp.get_group_disparity()
        best_noise = max(disparity.items(), key=lambda x: x[1])[0]
        sigma_at_05 = nsp.get_magnitude_at_threshold(best_noise, threshold=0.5, group=1)
        lines = [
            f"NSP diagnostic report — {nsp.model_name}",
            "=" * 60,
            "Vulnerability ranking (max flip rate across σ):",
            *[f"  {i + 1:>2}. {nt:<20s} {r:.3f}" for i, (nt, r) in enumerate(ranking)],
            "",
            "Per-noise group disparity (max over σ):",
            *[f"  {nt:<20s} {d:.2f}" for nt, d in
              sorted(disparity.items(), key=lambda x: x[1], reverse=True)],
            "",
            f"Recommended noise for CNI: {best_noise}",
            f"  magnitude at which minority flip rate >= 0.5: {sigma_at_05}",
        ]
        (output_dir / "nsp_summary.txt").write_text("\n".join(lines))
        return output_dir
