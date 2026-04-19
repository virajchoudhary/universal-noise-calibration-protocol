"""Calibrated Noise Injection — selects noise type + magnitude from an NSP.

Three calibration methods:

1. ``threshold`` — pick the noise with the highest per-group disparity; set
   σ_spurious to the smallest magnitude where the worst-group flip rate
   exceeds ``target_flip_rate`` and the best-group flip rate stays below
   ``best_group_cap``. If no magnitude satisfies both, ``confidence`` is
   downgraded but calibration still returns a reasonable σ.

2. ``mi_inspired`` — finds the *phase transition* in the NSP curve:
   the magnitude where the slope of the worst-group flip rate is maximal.
   Information-theoretic intuition: near the phase transition, I(spurious
   feature ; representation) drops sharply for the minority group while
   I(causal feature ; representation) is preserved for the majority.

3. ``adaptive`` — returns a schedule function that adjusts σ online
   based on the live train/val worst-group gap (see
   :class:`AdaptiveSigmaSchedule`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from uncp.nsa.sensitivity_probe import NoiseSensitivityProfile
from uncp.utils.io import load_pickle, save_pickle


@dataclass
class CalibrationConfig:
    recommended_noise_type: str
    sigma_spurious: float
    sigma_low: float
    sigma_high: float
    calibration_method: str
    confidence: float
    nsp_snapshot: Optional[NoiseSensitivityProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationConfig":
        return load_pickle(path)

    def describe(self) -> str:
        return (
            f"CNI calibration ({self.calibration_method})\n"
            f"  recommended noise : {self.recommended_noise_type}\n"
            f"  σ_low / σ_spurious / σ_high : "
            f"{self.sigma_low:.3f} / {self.sigma_spurious:.3f} / {self.sigma_high:.3f}\n"
            f"  confidence        : {self.confidence:.2f}\n"
            f"  metadata          : {self.metadata}"
        )

    def get_schedule(self, epoch: int, total_epochs: int) -> float:
        """Three-phase schedule mapped over [0, total_epochs).

        Equivalent to :class:`ThreePhaseSchedule` with default ratios
        (0.15 / 0.60 / 0.25); included for convenience / legacy use.
        """
        phase_a = max(1, int(total_epochs * 0.15))
        phase_b = max(1, int(total_epochs * 0.60))
        phase_c = max(1, total_epochs - phase_a - phase_b)
        if epoch < phase_a:
            return 0.0
        if epoch < phase_a + phase_b:
            prog = (epoch - phase_a) / phase_b
            return float(self.sigma_low + (self.sigma_high - self.sigma_low) *
                         (1 - np.cos(np.pi * prog)) / 2.0)
        k = epoch - phase_a - phase_b
        sigma_finetune = 0.1 * self.sigma_high
        return float(max(0.0, sigma_finetune * (1 - k / phase_c)))


class CNICalibrator:
    """Map an NSP → CalibrationConfig via one of three methods."""

    def __init__(
        self,
        nsp: NoiseSensitivityProfile,
        calibration_method: str = "threshold",
        target_flip_rate: float = 0.5,
        best_group_cap: float = 0.3,
    ) -> None:
        if calibration_method not in {"threshold", "mi_inspired", "adaptive"}:
            raise ValueError(f"unknown calibration method: {calibration_method}")
        self.nsp = nsp
        self.calibration_method = calibration_method
        self.target_flip_rate = float(target_flip_rate)
        self.best_group_cap = float(best_group_cap)

    # ------------------------------------------------------------------
    def _pick_noise_type(self) -> str:
        disparity = self.nsp.get_group_disparity()
        return max(disparity.items(), key=lambda kv: kv[1])[0]

    def _worst_best_groups(self) -> tuple[int, int]:
        """Identify which group is 'worst' (highest flip) and 'best' at peak σ."""
        any_per_mag = next(iter(self.nsp.results.values()))
        last_mag = max(any_per_mag.keys())
        pg = any_per_mag[last_mag]["per_group_flip"]
        if not pg:
            return 0, 0
        worst = max(pg, key=lambda g: pg[g])
        best = min(pg, key=lambda g: pg[g])
        return int(worst), int(best)

    # ------------------------------------------------------------------
    def _calibrate_threshold(self) -> CalibrationConfig:
        nt = self._pick_noise_type()
        worst_g, best_g = self._worst_best_groups()
        per_mag = self.nsp.results[nt]
        mags = sorted(per_mag.keys())

        sigma_spurious: Optional[float] = None
        for mag in mags:
            stats = per_mag[mag]
            pg = stats["per_group_flip"] or {}
            worst_flip = pg.get(worst_g, stats["flip_rate"])
            best_flip = pg.get(best_g, stats["flip_rate"])
            if worst_flip >= self.target_flip_rate and best_flip <= self.best_group_cap:
                sigma_spurious = float(mag)
                break

        confidence = 1.0
        if sigma_spurious is None:
            # Relaxed fallback: smallest σ with worst_flip ≥ target.
            for mag in mags:
                stats = per_mag[mag]
                pg = stats["per_group_flip"] or {}
                if pg.get(worst_g, stats["flip_rate"]) >= self.target_flip_rate:
                    sigma_spurious = float(mag)
                    break
            confidence = 0.6
        if sigma_spurious is None:
            # Ultimate fallback: midpoint of magnitude range.
            sigma_spurious = float(np.median(mags))
            confidence = 0.3

        sigma_low = max(0.01, 0.5 * sigma_spurious)
        sigma_high = min(1.0, 1.5 * sigma_spurious)

        disparity = self.nsp.get_group_disparity()
        meta = {
            "worst_group": worst_g, "best_group": best_g,
            "noise_disparity": disparity[nt],
            "all_disparities": disparity,
            "target_flip_rate": self.target_flip_rate,
            "best_group_cap": self.best_group_cap,
        }
        return CalibrationConfig(
            recommended_noise_type=nt,
            sigma_spurious=sigma_spurious,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            calibration_method="threshold",
            confidence=confidence,
            nsp_snapshot=self.nsp,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    def _calibrate_mi_inspired(self) -> CalibrationConfig:
        """Locate the phase transition in the worst-group flip-rate curve.

        Finite-difference derivative over σ; pick the σ at the steepest
        rise. This corresponds intuitively to the magnitude where
        MI(spurious feature; representation) drops sharply — the
        information-theoretic inflection point.
        """
        nt = self._pick_noise_type()
        worst_g, best_g = self._worst_best_groups()
        per_mag = self.nsp.results[nt]
        mags = np.array(sorted(per_mag.keys()))
        worst_flips = np.array([per_mag[float(m)]["per_group_flip"].get(worst_g,
                                per_mag[float(m)]["flip_rate"]) for m in mags])
        if len(mags) >= 2:
            deriv = np.gradient(worst_flips, mags)
            sigma_spurious = float(mags[int(np.argmax(deriv))])
        else:
            sigma_spurious = float(mags[0])

        sigma_low = max(0.01, 0.5 * sigma_spurious)
        sigma_high = min(1.0, 1.5 * sigma_spurious)

        disparity = self.nsp.get_group_disparity()
        meta = {
            "worst_group": worst_g, "best_group": best_g,
            "phase_transition_sigma": sigma_spurious,
            "noise_disparity": disparity[nt],
            "worst_flip_curve": worst_flips.tolist(),
            "magnitudes": mags.tolist(),
        }
        return CalibrationConfig(
            recommended_noise_type=nt,
            sigma_spurious=sigma_spurious,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            calibration_method="mi_inspired",
            confidence=0.8,
            nsp_snapshot=self.nsp,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    def _calibrate_adaptive(self) -> CalibrationConfig:
        base = self._calibrate_threshold()
        base.calibration_method = "adaptive"
        base.metadata["base_calibration"] = "threshold"
        base.metadata["adaptive"] = True
        return base

    # ------------------------------------------------------------------
    def calibrate(self) -> CalibrationConfig:
        if self.calibration_method == "threshold":
            return self._calibrate_threshold()
        if self.calibration_method == "mi_inspired":
            return self._calibrate_mi_inspired()
        return self._calibrate_adaptive()


class AdaptiveSigmaSchedule:
    """Closure-style schedule that adapts σ based on live worst-group accuracy.

    Usage::

        sched = AdaptiveSigmaSchedule(base=σ_spurious)
        σ_t = sched(epoch, worst_group_acc, avg_acc)
    """

    def __init__(self, base: float, up: float = 1.1, down: float = 0.9,
                 floor: float = 0.05, ceil: float = 1.0,
                 wga_target: float = 0.6, avg_floor: float = 0.5) -> None:
        self.base = float(base)
        self.up = float(up)
        self.down = float(down)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.wga_target = float(wga_target)
        self.avg_floor = float(avg_floor)
        self._current = self.base

    def __call__(self, epoch: int, wga: float, avg: float) -> float:
        if wga < self.wga_target:
            self._current = min(self.ceil, self._current * self.up)
        elif avg < self.avg_floor:
            self._current = max(self.floor, self._current * self.down)
        return self._current
