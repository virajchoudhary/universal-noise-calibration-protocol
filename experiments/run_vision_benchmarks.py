"""
Vision benchmark experiments for Phase 5.

Runs UNCP pipeline on:
1. Waterbirds (real or synthetic fallback)
2. CIFAR-10 with spurious watermark (at multiple rho values)

Usage:
    # Quick smoke test
    python experiments/run_vision_benchmarks.py --quick

    # Full run
    python experiments/run_vision_benchmarks.py

    # Specific benchmark only
    python experiments/run_vision_benchmarks.py --benchmark waterbirds
    python experiments/run_vision_benchmarks.py --benchmark cifar10
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from uncp.utils.seed import set_seed
from uncp.utils.io import save_json, ensure_dir
from uncp.nsa.sensitivity_probe import SensitivityProbe
from uncp.nsa.noise_generators import get_noise_generators
from uncp.cni.calibrator import CNICalibrator
from uncp.cni.noise_schedules import ThreePhaseSchedule


# ═══════════════════════════════════════════════════════════
# Model Factories
# ═══════════════════════════════════════════════════════════

def create_resnet18_waterbirds(pretrained=True):
    """ResNet-18 for Waterbirds (224x224, 2 classes)."""
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def create_resnet18_cifar10():
    """ResNet-18 for CIFAR-10 (32x32, 10 classes)."""
    from torchvision.models import resnet18
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ═══════════════════════════════════════════════════════════
# Shared Training / Eval Utilities
# ═══════════════════════════════════════════════════════════

def _quick_eval(model, loader, device):
    """Quick evaluation returning accuracy, wga, per_group."""
    model.eval()
    correct = 0
    total = 0
    group_correct = {}
    group_total = {}

    with torch.no_grad():
        for batch in loader:
            if len(batch) >= 3:
                x, y, g = batch[0], batch[1], batch[2]
            else:
                x, y = batch[0], batch[1]
                g = torch.zeros_like(y)

            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            for i in range(x.size(0)):
                gi = g[i].item()
                group_correct[gi] = group_correct.get(gi, 0) + int(preds[i] == y[i])
                group_total[gi] = group_total.get(gi, 0) + 1

    accuracy = correct / max(total, 1)
    per_group = {gi: group_correct[gi] / max(group_total[gi], 1) for gi in sorted(group_total)}
    wga = min(per_group.values()) if per_group else 0.0

    return {"accuracy": accuracy, "wga": wga, "per_group": per_group}


def train_erm(model, train_loader, val_loader, epochs, lr, weight_decay, device, verbose=True):
    """Standard ERM training. Returns trained model and val metrics history."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_wga = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if len(batch) >= 3:
                x, y = batch[0], batch[1]
            else:
                x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_metrics = _quick_eval(model, val_loader, device)
        if val_metrics["wga"] > best_wga:
            best_wga = val_metrics["wga"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}/{epochs}: val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_wga={val_metrics['wga']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_wga


def run_uncp_on_benchmark(
    benchmark_name,
    model_factory,
    train_loader,
    val_loader,
    test_loader,
    epochs=10,
    nsa_samples=500,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    """Run the full UNCP pipeline on a given benchmark.

    Steps:
    1. Train ERM warm-start
    2. Run NSA diagnostic
    3. Run CNI calibration
    4. Three-phase retraining
    5. Post-UNCP NSA
    6. Evaluate

    Returns dict with all results.
    """
    print(f"\n{'='*60}")
    print(f"UNCP PIPELINE: {benchmark_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Stage 1: Warm-start ERM
    print(f"\n[{benchmark_name}] Stage 0: Warm-start ERM...")
    warmstart_epochs = max(2, epochs // 4)
    model = model_factory()
    model, erm_wga = train_erm(
        model, train_loader, val_loader,
        epochs=warmstart_epochs, lr=lr, weight_decay=weight_decay, device=device,
    )
    print(f"  ERM warm-start WGA: {erm_wga:.4f}")

    # Full ERM baseline (separate model)
    print(f"\n[{benchmark_name}] Training full ERM baseline...")
    erm_model = model_factory()
    erm_model, erm_full_wga = train_erm(
        erm_model, train_loader, val_loader,
        epochs=epochs, lr=lr, weight_decay=weight_decay, device=device,
    )

    # Stage 2: NSA
    print(f"\n[{benchmark_name}] Stage 1: NSA diagnostic...")
    noise_gens = get_noise_generators("vision")
    probe = SensitivityProbe(
        model=erm_model,
        noise_generators=noise_gens,
        num_samples=min(nsa_samples, len(test_loader.dataset)),
        device=device,
    )
    nsp_before = probe.probe(test_loader)
    print(f"  Most sensitive: {nsp_before.get_most_sensitive_noise()}")
    print(f"  Ranking: {nsp_before.get_vulnerability_ranking()[:3]}")

    # Stage 3: CNI
    print(f"\n[{benchmark_name}] Stage 2: CNI calibration...")
    calibrator = CNICalibrator(nsp=nsp_before, calibration_method="threshold", target_flip_rate=0.5)
    calib = calibrator.calibrate()
    print(f"  Noise type: {calib.recommended_noise_type}, sigma={calib.sigma_spurious:.4f}")

    # Stage 4: Retrain
    print(f"\n[{benchmark_name}] Stage 3: Three-phase retraining...")
    phase_a = max(1, epochs // 5)
    phase_c = max(1, epochs // 5)
    phase_b = epochs - phase_a - phase_c

    schedule = ThreePhaseSchedule(
        total_epochs=epochs,
        phase_a_epochs=phase_a,
        phase_b_epochs=phase_b,
        phase_c_epochs=phase_c,
        sigma_low=calib.sigma_low,
        sigma_high=calib.sigma_high,
        annealing_strategy="cosine",
    )

    rec_noise_gen = noise_gens[calib.recommended_noise_type]
    uncp_model = model_factory().to(device)
    optimizer = torch.optim.AdamW(uncp_model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_wga = 0.0
    best_state = None

    for epoch in range(epochs):
        uncp_model.train()
        magnitude = schedule.get_magnitude(epoch)
        phase = schedule.get_phase(epoch)

        for batch in train_loader:
            if len(batch) >= 3:
                x, y = batch[0], batch[1]
            else:
                x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            if magnitude > 0:
                x = rec_noise_gen.apply(x, magnitude)

            logits = uncp_model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sched.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            val_m = _quick_eval(uncp_model, val_loader, device)
            print(f"    Epoch {epoch+1}/{epochs} [{phase}] sigma={magnitude:.3f}: "
                  f"val_wga={val_m['wga']:.4f}")
            if val_m["wga"] > best_wga:
                best_wga = val_m["wga"]
                best_state = {k: v.cpu().clone() for k, v in uncp_model.state_dict().items()}

    if best_state is not None:
        uncp_model.load_state_dict(best_state)
        uncp_model.to(device)

    # Stage 5: Post-UNCP NSA
    print(f"\n[{benchmark_name}] Stage 4: Post-UNCP NSA...")
    probe_after = SensitivityProbe(
        model=uncp_model,
        noise_generators=noise_gens,
        num_samples=min(nsa_samples, len(test_loader.dataset)),
        device=device,
    )
    nsp_after = probe_after.probe(test_loader)

    # Final evaluation
    erm_test = _quick_eval(erm_model, test_loader, device)
    uncp_test = _quick_eval(uncp_model, test_loader, device)

    elapsed = time.time() - start_time

    results = {
        "benchmark": benchmark_name,
        "erm_id_average_accuracy": erm_test["accuracy"],
        "erm_ood_worst_group_accuracy": erm_test["wga"],
        "erm_ood_per_group_accuracy": erm_test["per_group"],
        "uncp_id_average_accuracy": uncp_test["accuracy"],
        "uncp_ood_worst_group_accuracy": uncp_test["wga"],
        "uncp_ood_per_group_accuracy": uncp_test["per_group"],
        "calibrated_noise_type": calib.recommended_noise_type,
        "calibrated_sigma": calib.sigma_spurious,
        "nsp_before_most_sensitive": nsp_before.get_most_sensitive_noise(),
        "nsp_after_most_sensitive": nsp_after.get_most_sensitive_noise(),
        "schedule_phases": f"{phase_a}+{phase_b}+{phase_c}",
        "training_time_seconds": elapsed,
        "epochs": epochs,
    }

    print(f"\n[{benchmark_name}] RESULTS:")
    print(f"  ERM:  ID Acc={erm_test['accuracy']:.4f}, OOD WGA={erm_test['wga']:.4f}")
    print(f"  UNCP: ID Acc={uncp_test['accuracy']:.4f}, OOD WGA={uncp_test['wga']:.4f}")
    print(f"  Delta WGA: {uncp_test['wga'] - erm_test['wga']:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Vision Benchmarks")
    parser.add_argument("--benchmark", choices=["waterbirds", "cifar10", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="Smoke test with tiny epochs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nsa-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"

    if args.quick:
        epochs = args.epochs or 3
        nsa_samples = min(args.nsa_samples, 50)
    else:
        epochs = args.epochs or 10
        nsa_samples = args.nsa_samples

    all_results = []

    # ── Waterbirds ──
    if args.benchmark in ("waterbirds", "both"):
        print("\n" + "=" * 60)
        print("BENCHMARK: WATERBIRDS")
        print("=" * 60)

        from uncp.data.waterbirds import get_waterbirds_dataloaders

        config = OmegaConf.create({
            "dataset": {"root": "./data/waterbirds"},
            "training": {"batch_size": 64},
        })

        train_loader, val_loader, test_loader = get_waterbirds_dataloaders(
            config, use_fallback=True
        )

        results = run_uncp_on_benchmark(
            benchmark_name="waterbirds",
            model_factory=create_resnet18_waterbirds,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            nsa_samples=nsa_samples,
            lr=1e-4,  # Lower LR for pretrained model
            weight_decay=1e-4,
            device=device,
        )
        all_results.append(results)

    # ── CIFAR-10 Watermark ──
    if args.benchmark in ("cifar10", "both"):
        print("\n" + "=" * 60)
        print("BENCHMARK: CIFAR-10 WATERMARK")
        print("=" * 60)

        from uncp.data.cifar10_watermark import get_cifar10_watermark_dataloaders

        # Run at multiple correlation strengths
        for rho in [0.9] if args.quick else [0.7, 0.9, 1.0]:
            config = OmegaConf.create({
                "dataset": {
                    "root": "./data",
                    "correlation_strength": rho,
                    "batch_size": 128,
                },
                "training": {"batch_size": 128},
            })

            train_loader, val_loader, test_loader, clean_loader = \
                get_cifar10_watermark_dataloaders(config)

            results = run_uncp_on_benchmark(
                benchmark_name=f"cifar10_watermark_rho{rho}",
                model_factory=create_resnet18_cifar10,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                nsa_samples=nsa_samples,
                lr=1e-3,
                weight_decay=1e-4,
                device=device,
            )
            results["rho"] = rho
            all_results.append(results)

    # ── Save all results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / f"vision_benchmarks_{timestamp}"
    ensure_dir(str(output_dir))
    save_json(all_results, str(output_dir / "results.json"))

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<30} {'ERM WGA':>10} {'UNCP WGA':>10} {'Delta':>10}")
    print("-" * 62)
    for r in all_results:
        name = r["benchmark"]
        erm_wga = r.get("erm_ood_worst_group_accuracy", 0)
        uncp_wga = r.get("uncp_ood_worst_group_accuracy", 0)
        delta = uncp_wga - erm_wga
        print(f"{name:<30} {erm_wga:>10.4f} {uncp_wga:>10.4f} {delta:>+10.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()