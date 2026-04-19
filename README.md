# UNCP — Universal Noise Calibration Protocol

A domain-agnostic framework for mitigating spurious correlations in deep learning
via **diagnosed, targeted noise injection**. Differs from random augmentation (Mixup,
CutMix) and label-requiring methods (Group DRO) by using a diagnostic stage (NSA)
to *select* which noise to inject, then calibrating its magnitude to break spurious
features while preserving causal ones.

## Pipeline

1. **NSA — Noise Sensitivity Analysis.** Probe a trained model with many noise
   types at many magnitudes and record per-group flip rates. The noise with the
   highest group disparity targets the spurious feature.
2. **CNI — Calibrated Noise Injection.** Use the NSP to pick a noise type and a
   magnitude `σ` where worst-group flip rate exceeds 0.5 but best-group flip
   rate stays below 0.3.
3. **Three-Phase Retraining.** Warm-up (no noise) → injection (annealed σ_low → σ_high)
   → fine-tune (residual noise decaying to 0).
4. **SRD Validation.** Spurious Robustness Degradation metric + ΔNSP.

## Setup

```
pip install -r requirements.txt
```

## Repro — primary result

```
python experiments/run_baselines.py          # Phase 1 ERM baseline
python experiments/run_nsa.py                # Phase 2 NSA diagnostic
python experiments/run_full_pipeline.py      # Phase 3 full UNCP
python experiments/run_comparison.py         # Phase 4 full method comparison
```

## Directory layout

```
UNCP/
├── configs/         {vision,nlp,tabular}/*.yaml
├── uncp/
│   ├── nsa/         noise_generators, sensitivity_probe, nsp_visualizer
│   ├── cni/         calibrator, noise_schedules
│   ├── training/    trainer (pipeline), phases, callbacks
│   ├── evaluation/  srd, metrics, reporting
│   ├── data/        per-dataset loaders
│   └── utils/       seed, io
├── baselines/       erm, group_dro, jtt, mixup, cutmix, adversarial, dropout
├── experiments/     entry-point scripts
├── results/         saved JSON + figures per run
├── checkpoints/
└── paper/           figures/, tables/
```
