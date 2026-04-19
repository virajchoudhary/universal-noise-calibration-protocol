"""Three-phase noise injection schedule."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .calibrator import CalibrationConfig


@dataclass
class ThreePhaseSchedule:
    """Warm-up → annealed injection → fine-tune decay.

    Parameters
    ----------
    total_epochs : int
    phase_a_epochs : int
        No-noise warm-up.
    phase_b_epochs : int
        Annealed noise injection (σ_low → σ_high).
    phase_c_epochs : int
        Fine-tune with residual noise decaying to 0.
    sigma_low, sigma_high : float
    sigma_finetune : Optional[float]
        Peak residual noise in Phase C; defaults to 0.1 · sigma_high.
    annealing_strategy : {"linear", "cosine", "step"}
    """

    total_epochs: int
    phase_a_epochs: int
    phase_b_epochs: int
    phase_c_epochs: int
    sigma_low: float
    sigma_high: float
    sigma_finetune: Optional[float] = None
    annealing_strategy: str = "cosine"

    def __post_init__(self) -> None:
        if self.sigma_finetune is None:
            self.sigma_finetune = 0.1 * self.sigma_high
        total = self.phase_a_epochs + self.phase_b_epochs + self.phase_c_epochs
        if total != self.total_epochs:
            raise ValueError(
                f"phase epochs ({total}) must sum to total_epochs ({self.total_epochs})"
            )

    # ------------------------------------------------------------------
    def get_phase(self, epoch: int) -> str:
        if epoch < self.phase_a_epochs:
            return "warm_up"
        if epoch < self.phase_a_epochs + self.phase_b_epochs:
            return "noise_injection"
        return "fine_tuning"

    def get_magnitude(self, epoch: int) -> float:
        if epoch < self.phase_a_epochs:
            return 0.0
        if epoch < self.phase_a_epochs + self.phase_b_epochs:
            prog = (epoch - self.phase_a_epochs) / max(1, self.phase_b_epochs - 1 or 1)
            prog = float(min(max(prog, 0.0), 1.0))
            if self.annealing_strategy == "linear":
                return float(self.sigma_low + (self.sigma_high - self.sigma_low) * prog)
            if self.annealing_strategy == "cosine":
                return float(
                    self.sigma_low
                    + (self.sigma_high - self.sigma_low) * (1 - math.cos(math.pi * prog)) / 2.0
                )
            if self.annealing_strategy == "step":
                return self.sigma_low if prog < 0.5 else self.sigma_high
            raise ValueError(f"unknown annealing: {self.annealing_strategy}")
        k = epoch - self.phase_a_epochs - self.phase_b_epochs
        prog = k / max(1, self.phase_c_epochs)
        return float(max(0.0, self.sigma_finetune * (1.0 - prog)))

    # ------------------------------------------------------------------
    def get_config_snapshot(self) -> dict:
        return {
            "total_epochs": self.total_epochs,
            "phase_a_epochs": self.phase_a_epochs,
            "phase_b_epochs": self.phase_b_epochs,
            "phase_c_epochs": self.phase_c_epochs,
            "sigma_low": self.sigma_low,
            "sigma_high": self.sigma_high,
            "sigma_finetune": self.sigma_finetune,
            "annealing_strategy": self.annealing_strategy,
        }

    # ------------------------------------------------------------------
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        epochs = np.arange(self.total_epochs)
        sigmas = np.array([self.get_magnitude(int(e)) for e in epochs])

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(epochs, sigmas, color="black", lw=1.5)
        a, b = self.phase_a_epochs, self.phase_a_epochs + self.phase_b_epochs
        ax.axvspan(0, a, color="#d0e8ff", alpha=0.7, label="A: warm-up")
        ax.axvspan(a, b, color="#fff4c2", alpha=0.7, label="B: injection")
        ax.axvspan(b, self.total_epochs, color="#d6f5d6", alpha=0.7,
                   label="C: fine-tune")
        ax.set_xlabel("epoch")
        ax.set_ylabel("noise magnitude σ")
        ax.set_title(f"Three-phase schedule ({self.annealing_strategy} anneal)")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig


def create_schedule_from_config(
    calibration_config: CalibrationConfig,
    total_epochs: int,
    phase_ratios: tuple[float, float, float] = (0.15, 0.60, 0.25),
    annealing: str = "cosine",
) -> ThreePhaseSchedule:
    ra, rb, rc = phase_ratios
    a = max(1, int(total_epochs * ra))
    b = max(1, int(total_epochs * rb))
    c = max(1, total_epochs - a - b)
    return ThreePhaseSchedule(
        total_epochs=a + b + c,
        phase_a_epochs=a, phase_b_epochs=b, phase_c_epochs=c,
        sigma_low=calibration_config.sigma_low,
        sigma_high=calibration_config.sigma_high,
        annealing_strategy=annealing,
    )
