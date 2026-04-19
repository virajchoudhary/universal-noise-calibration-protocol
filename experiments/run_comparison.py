"""
Colored MNIST Full Comparison Experiment (Step 4.2).

Runs all baseline methods and UNCP on Colored MNIST with rho=0.9,
producing the paper's main results table, LaTeX output, and
comparison figures.

Usage:
    # Quick smoke test (tiny epochs, subset of data)
    python experiments/run_comparison.py --quick

    # Full comparison (all methods, full epochs)
    python experiments/run_comparison.py

    # Select specific methods
    python experiments/run_comparison.py --methods erm,jtt,uncp

    # Reuse existing UNCP results (skip re-running UNCP pipeline)
    python experiments/run_comparison.py --reuse-existing-uncp results/uncp_colored_mnist_*/

    # Override epoch count
    python experiments/run_comparison.py --epochs 5 --uncp-epochs 8

    # Override NSA probe samples
    python experiments/run_comparison.py --nsa-samples 200
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

# -- Add project root to path --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from uncp.utils.seed import set_seed
from uncp.utils.io import save_json
from uncp.data.colored_mnist import ColoredMNIST, get_colored_mnist_dataloaders
from uncp.evaluation.srd import SRDCalculator
from uncp.nsa.sensitivity_probe import SensitivityProbe
from uncp.nsa.noise_generators import get_noise_generators
from uncp.cni.calibrator import CNICalibrator
from uncp.cni.noise_schedules import ThreePhaseSchedule
from baselines.registry import get_baseline, get_baseline_names, NO_GROUP_LABEL_METHODS


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def unpack_batch(batch):
    """Support this repo's dict batches and tuple-style batches."""
    if isinstance(batch, dict):
        x = batch["image"]
        y = batch["label"]
        g = batch.get("group_label", torch.zeros_like(y))
        return x, y, g
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    x, y = batch[0], batch[1]
    return x, y, torch.zeros_like(y)


# ==========================================================
# Model Factory
# ==========================================================

def create_resnet18_colored_mnist() -> nn.Module:
    """Create ResNet-18 adapted for Colored MNIST (28x28, 10 classes).

    Modifications from standard ResNet-18:
    - First conv: 3x3, stride 1 (instead of 7x7, stride 2)
    - Remove first MaxPool
    - Final FC: 512 -> 10
    """
    from torchvision.models import resnet18

    model = resnet18(num_classes=10)
    # Adapt first conv for 28x28 images
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove maxpool for small images
    model.maxpool = nn.Identity()
    return model


# ==========================================================
# SRD Computation Helper
# ==========================================================

def compute_srd_for_model(
    model: nn.Module,
    id_test_loader,
    ood_test_loader,
    device: str = "cpu",
) -> dict:
    """Compute SRD for a trained model using ID and OOD test loaders.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    id_test_loader : DataLoader
        In-distribution test set (rho=0.9 for Colored MNIST).
    ood_test_loader : DataLoader
        Out-of-distribution test set (rho=0.0 for Colored MNIST).
    device : str
        Device string.

    Returns
    -------
    dict with srd, srd_v2, per_group_clean_acc, per_group_corrupt_acc, etc.
    """
    try:
        srd_calc = SRDCalculator(
            model=model,
            clean_loader=id_test_loader,
            corrupted_loader=ood_test_loader,
            device=device,
        )
        result = srd_calc.compute()
        return {
            "srd": result.srd,
            "srd_v2": result.srd_v2,
            "per_group_clean_acc": result.per_group_clean_acc,
            "per_group_corrupt_acc": result.per_group_corrupt_acc,
            "per_group_degradation": result.per_group_degradation,
        }
    except Exception as e:
        print(f"  [SRD] Computation failed: {e}")
        return {
            "srd": None,
            "srd_v2": None,
            "per_group_clean_acc": {},
            "per_group_corrupt_acc": {},
            "per_group_degradation": {},
            "srd_error": str(e),
        }


# ==========================================================
# UNCP Pipeline Runner
# ==========================================================

def run_uncp_pipeline(
    train_loader,
    val_loader,
    id_test_loader,
    ood_test_loader,
    config,
    device: str = "cpu",
    nsa_samples: int = 500,
    uncp_epochs: int = 10,
) -> dict:
    """Run the full UNCP pipeline: NSA -> CNI -> Retrain -> Validate.

    Returns
    -------
    dict with all UNCP results including NSP comparison.
    """
    from baselines.erm import ERMTrainer

    print("\n" + "=" * 60)
    print("UNCP PIPELINE")
    print("=" * 60)

    # -- Stage 1: Warm-start ERM --
    print("\n[UNCP Stage 0] Warm-start ERM training...")
    warmstart_epochs = max(2, uncp_epochs // 5)
    warmstart_config = copy.deepcopy(config)
    OmegaConf.update(warmstart_config, "training.epochs", warmstart_epochs)

    model = create_resnet18_colored_mnist().to(device)
    erm_trainer = ERMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=ood_test_loader,
        config=warmstart_config,
        device=device,
    )
    erm_results = erm_trainer.run()
    erm_model = erm_trainer.model

    # -- Stage 2: NSA Diagnostic --
    print("\n[UNCP Stage 1] Noise Sensitivity Analysis...")
    noise_gens = get_noise_generators("vision")
    probe = SensitivityProbe(
        model=erm_model,
        noise_generators=noise_gens,
        num_samples=min(nsa_samples, len(ood_test_loader.dataset)),
        device=device,
    )
    nsp_before = probe.probe(ood_test_loader)

    print(f"  Most sensitive: {nsp_before.get_most_sensitive_noise()}")
    print(f"  Vulnerability ranking: {nsp_before.get_vulnerability_ranking()}")

    # -- Stage 3: CNI Calibration --
    print("\n[UNCP Stage 2] Calibrated Noise Injection setup...")
    calibrator = CNICalibrator(
        nsp=nsp_before,
        calibration_method="threshold",
        target_flip_rate=0.5,
    )
    calib_config = calibrator.calibrate()

    print(f"  Recommended noise: {calib_config.recommended_noise_type}")
    print(f"  sigma_spurious: {calib_config.sigma_spurious:.4f}")
    print(f"  sigma range: [{calib_config.sigma_low:.4f}, {calib_config.sigma_high:.4f}]")

    # -- Stage 4: Three-Phase Retraining --
    print("\n[UNCP Stage 3] Three-phase retraining with calibrated noise...")
    phase_a = max(1, uncp_epochs // 5)
    phase_c = max(1, uncp_epochs // 5)
    phase_b = uncp_epochs - phase_a - phase_c

    schedule = ThreePhaseSchedule(
        total_epochs=uncp_epochs,
        phase_a_epochs=phase_a,
        phase_b_epochs=phase_b,
        phase_c_epochs=phase_c,
        sigma_low=calib_config.sigma_low,
        sigma_high=calib_config.sigma_high,
        annealing_strategy="cosine",
    )

    # Get the recommended noise generator
    rec_noise_type = calib_config.recommended_noise_type
    rec_noise_gen = noise_gens[rec_noise_type]

    # Retrain from scratch with noise
    uncp_model = create_resnet18_colored_mnist().to(device)
    optimizer = torch.optim.AdamW(
        uncp_model.parameters(),
        lr=config.training.lr if hasattr(config, "training") else 1e-3,
        weight_decay=config.training.weight_decay if hasattr(config, "training") else 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=uncp_epochs)

    best_wga = 0.0
    best_state = None

    for epoch in range(uncp_epochs):
        uncp_model.train()
        magnitude = schedule.get_magnitude(epoch)
        phase = schedule.get_phase(epoch)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            x, y, _ = unpack_batch(batch)

            x = x.to(device)
            y = y.to(device)

            # Apply noise if in Phase B or C
            if magnitude > 0:
                x = rec_noise_gen.apply(x, magnitude)

            logits = uncp_model(x)
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_total += x.size(0)

        scheduler.step()

        # Evaluate
        if (epoch + 1) % max(1, uncp_epochs // 5) == 0 or epoch == 0:
            uncp_model.eval()
            val_metrics = _quick_eval(uncp_model, val_loader, device)
            print(
                f"  Epoch {epoch+1}/{uncp_epochs} [{phase}] "
                f"sigma={magnitude:.3f}: "
                f"train_acc={epoch_correct/max(epoch_total,1):.4f}, "
                f"val_wga={val_metrics['wga']:.4f}"
            )
            if val_metrics["wga"] > best_wga:
                best_wga = val_metrics["wga"]
                best_state = {k: v.cpu().clone() for k, v in uncp_model.state_dict().items()}

    # Load best model
    if best_state is not None:
        uncp_model.load_state_dict(best_state)
        uncp_model.to(device)

    # -- Stage 5: Post-UNCP NSA --
    print("\n[UNCP Stage 4] Post-treatment NSA...")
    probe_after = SensitivityProbe(
        model=uncp_model,
        noise_generators=noise_gens,
        num_samples=min(nsa_samples, len(ood_test_loader.dataset)),
        device=device,
    )
    nsp_after = probe_after.probe(ood_test_loader)

    # Compute delta NSP
    ranking_before = dict(nsp_before.get_vulnerability_ranking())
    ranking_after = dict(nsp_after.get_vulnerability_ranking())
    delta_nsp = {}
    for noise_type in ranking_before:
        delta_nsp[noise_type] = ranking_before.get(noise_type, 0) - ranking_after.get(noise_type, 0)

    # Final evaluation
    test_metrics = _quick_eval(uncp_model, ood_test_loader, device)
    id_metrics = _quick_eval(uncp_model, id_test_loader, device)

    results = {
        "method": "uncp",
        "id_average_accuracy": id_metrics["accuracy"],
        "id_worst_group_accuracy": id_metrics["wga"],
        "id_per_group_accuracy": id_metrics["per_group"],
        "ood_average_accuracy": test_metrics["accuracy"],
        "ood_worst_group_accuracy": test_metrics["wga"],
        "ood_per_group_accuracy": test_metrics["per_group"],
        "val_average_accuracy": _quick_eval(uncp_model, val_loader, device)["accuracy"],
        "val_worst_group_accuracy": _quick_eval(uncp_model, val_loader, device)["wga"],
        "srd": None,
        "srd_v2": None,
        "training_time_seconds": 0,  # Will be set by caller
        "requires_group_labels": False,
        "num_hyperparameters": 4,  # noise_type, sigma, schedule ratios, calibration method
        "relative_training_cost": "2x",
        "nsp_before_most_sensitive": nsp_before.get_most_sensitive_noise(),
        "nsp_after_most_sensitive": nsp_after.get_most_sensitive_noise(),
        "delta_nsp": delta_nsp,
        "calibrated_noise_type": rec_noise_type,
        "calibrated_sigma": calib_config.sigma_spurious,
        "schedule_phases": f"{phase_a}+{phase_b}+{phase_c}",
    }

    print(f"\n[UNCP] Final Results:")
    print(f"  ID Average accuracy:   {results['id_average_accuracy']:.4f}")
    print(f"  OOD Worst-group acc:   {results['ood_worst_group_accuracy']:.4f}")
    print(f"  OOD Per-group:         {results['ood_per_group_accuracy']}")
    print(f"  Calibrated noise:      {rec_noise_type} @ sigma={calib_config.sigma_spurious:.3f}")

    return results, nsp_before, nsp_after, uncp_model


def _quick_eval(model, loader, device):
    """Quick evaluation returning accuracy, wga, per_group."""
    model.eval()
    correct = 0
    total = 0
    group_correct = {}
    group_total = {}

    with torch.no_grad():
        for batch in loader:
            x, y, g = unpack_batch(batch)

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


def normalize_baseline_result(
    method_name: str,
    raw_result: dict,
    trainer,
    id_test_loader,
    ood_test_loader,
    device: str,
) -> dict:
    """Convert heterogeneous baseline trainer outputs into comparison rows."""
    model = trainer.model
    id_metrics = _quick_eval(model, id_test_loader, device)
    ood_metrics = _quick_eval(model, ood_test_loader, device)

    normalized = dict(raw_result) if isinstance(raw_result, dict) else {}
    normalized["method"] = normalized.get("method", method_name)

    normalized["ood_average_accuracy"] = normalized.pop(
        "average_accuracy", ood_metrics["accuracy"]
    )
    normalized["ood_worst_group_accuracy"] = normalized.pop(
        "worst_group_accuracy", ood_metrics["wga"]
    )
    normalized["ood_per_group_accuracy"] = normalized.pop(
        "per_group_accuracy", ood_metrics["per_group"]
    )
    normalized["id_average_accuracy"] = id_metrics["accuracy"]
    normalized["id_worst_group_accuracy"] = id_metrics["wga"]
    normalized["id_per_group_accuracy"] = id_metrics["per_group"]

    normalized.update({
        "requires_group_labels": method_name == "group_dro",
        "num_hyperparameters": normalized.get("num_hyperparameters", {
            "erm": 2,
            "group_dro": 3,
            "jtt": 4,
            "mixup": 3,
            "cutmix": 3,
            "adversarial": 3,
        }.get(method_name, 2)),
        "relative_training_cost": normalized.get("relative_training_cost", {
            "jtt": "2x",
            "adversarial": "11x",
        }.get(method_name, "1x")),
    })

    srd_result = compute_srd_for_model(model, id_test_loader, ood_test_loader, device)
    normalized["srd"] = srd_result.get("srd")
    normalized["srd_v2"] = srd_result.get("srd_v2")
    normalized["per_group_clean_acc"] = srd_result.get("per_group_clean_acc", {})
    normalized["per_group_corrupt_acc"] = srd_result.get("per_group_corrupt_acc", {})
    normalized["per_group_degradation"] = srd_result.get("per_group_degradation", {})
    return normalized


# ==========================================================
# Results Formatting
# ==========================================================

def format_main_results_table(results_list: list) -> str:
    """Generate LaTeX table for main results.

    Parameters
    ----------
    results_list : list of dict
        One dict per method with keys: method, id_average_accuracy,
        ood_worst_group_accuracy, ood_per_group_accuracy, srd, requires_group_labels.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Sort by worst-group accuracy (descending)
    sorted_results = sorted(
        results_list,
        key=lambda r: r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0)),
        reverse=True,
    )

    # Find best and second-best WGA
    wgas = [
        r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0))
        for r in sorted_results
    ]
    best_wga = max(wgas) if wgas else 0
    second_best_wga = sorted(set(wgas), reverse=True)[1] if len(set(wgas)) > 1 else 0

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Colored MNIST ($\rho=0.9$) comparison. Methods marked with * require group labels during training.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Group Labels? & ID Avg Acc & OOD WGA & SRD & Rel. Cost \\",
        r"\midrule",
    ]

    for r in sorted_results:
        method = r["method"].replace("_", "\\_")
        if r.get("requires_group_labels", False):
            method += "*"

        avg = r.get("id_average_accuracy", r.get("average_accuracy", 0))
        wga = r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0))
        srd = r.get("srd", None)
        cost = r.get("relative_training_cost", "1x")

        avg_str = f"{avg:.3f}"
        wga_str = f"{wga:.3f}"
        srd_str = f"{srd:.3f}" if srd is not None else "---"
        cost_str = str(cost)

        # Bold best, underline second best
        if abs(wga - best_wga) < 1e-6:
            wga_str = r"\textbf{" + wga_str + "}"
        elif abs(wga - second_best_wga) < 1e-6:
            wga_str = r"\underline{" + wga_str + "}"

        lines.append(f"{method} & {'Yes' if r.get('requires_group_labels') else 'No'} & {avg_str} & {wga_str} & {srd_str} & {cost_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_method_comparison_table(results_list: list) -> str:
    """Generate LaTeX method-comparison table (qualitative properties)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Method comparison: properties and requirements.}",
        r"\label{tab:method_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Group Labels & \# HPs & Training Cost \\",
        r"\midrule",
    ]

    for r in sorted(
        results_list,
        key=lambda r: r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0)),
        reverse=True,
    ):
        method = r["method"].replace("_", "\\_")
        gl = "Yes" if r.get("requires_group_labels") else "No"
        nhp = r.get("num_hyperparameters", "?")
        cost = r.get("relative_training_cost", "1x")
        lines.append(f"{method} & {gl} & {nhp} & {cost} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def plot_wga_comparison(results_list: list, save_path: str):
    """Bar chart comparing worst-group accuracy across methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sorted_results = sorted(
        results_list,
        key=lambda r: r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0)),
        reverse=True,
    )

    methods = [r["method"] for r in sorted_results]
    wgas = [
        r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0)) * 100
        for r in sorted_results
    ]

    # Color: UNCP gets a highlight color
    colors = []
    for m in methods:
        if m == "uncp":
            colors.append("#E74C3C")  # Red highlight
        elif m.startswith("group_dro"):
            colors.append("#95A5A6")  # Gray (oracle)
        else:
            colors.append("#3498DB")  # Blue

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(methods)), wgas, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Worst-Group Accuracy (%)", fontsize=12)
    ax.set_title("Colored MNIST (rho=0.9) - Worst-Group Accuracy Comparison", fontsize=13)
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Random (10%)")
    ax.legend(fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, wgas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  WGA bar chart saved to {save_path}")


def plot_srd_comparison(results_list: list, save_path: str):
    """Bar chart comparing SRD across methods (lower is better)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter methods with valid SRD
    valid = [r for r in results_list if r.get("srd") is not None]
    if not valid:
        print("  [SRD Plot] No valid SRD values - skipping plot.")
        return

    sorted_results = sorted(valid, key=lambda r: r.get("srd", float("inf")))

    methods = [r["method"] for r in sorted_results]
    srds = [r["srd"] for r in sorted_results]

    colors = []
    for m in methods:
        if m == "uncp":
            colors.append("#E74C3C")
        else:
            colors.append("#3498DB")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(methods)), srds, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("SRD (lower is better)", fontsize=12)
    ax.set_title("Colored MNIST (rho=0.9) - Spurious Robustness Degradation", fontsize=13)

    for bar, val in zip(bars, srds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SRD bar chart saved to {save_path}")


# ==========================================================
# Main Runner
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Colored MNIST Full Comparison (Step 4.2)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: use tiny epochs and subset of data",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="erm,group_dro,jtt,mixup,cutmix,adversarial,dropout_0.1,dropout_0.3,dropout_0.5,uncp",
        help="Comma-separated list of methods to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs for baselines",
    )
    parser.add_argument(
        "--uncp-epochs",
        type=int,
        default=None,
        help="Override UNCP retraining epochs",
    )
    parser.add_argument(
        "--nsa-samples",
        type=int,
        default=500,
        help="Number of samples for NSA probe",
    )
    parser.add_argument(
        "--warmstart",
        type=int,
        default=None,
        help="Override warm-start ERM epochs for UNCP",
    )
    parser.add_argument(
        "--reuse-existing-uncp",
        type=str,
        default=None,
        help="Path to existing UNCP results directory to skip re-running UNCP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # -- Configuration --
    set_seed(args.seed)
    device = torch.device("cpu")

    # Quick mode settings
    if args.quick:
        baseline_epochs = args.epochs or 2
        uncp_epochs = args.uncp_epochs or 3
        nsa_samples = min(args.nsa_samples, 50)
        train_subset = 500
        batch_size = 64
    else:
        baseline_epochs = args.epochs or 10
        uncp_epochs = args.uncp_epochs or 10
        nsa_samples = args.nsa_samples
        train_subset = None
        batch_size = 128

    method_list = [m.strip() for m in args.methods.split(",")]

    # Load config
    config_path = PROJECT_ROOT / "configs" / "vision" / "colored_mnist.yaml"
    if config_path.exists():
        config = OmegaConf.load(str(config_path))
    else:
        config = OmegaConf.create({
            "seed": args.seed,
            "training": {
                "epochs": baseline_epochs,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": batch_size,
            },
            "dataset": {
                "name": "colored_mnist",
                "correlation_strength": 0.9,
                "label_noise": 0.25,
            },
        })

    # Override epochs in config
    OmegaConf.update(config, "training.epochs", baseline_epochs)
    OmegaConf.update(config, "training.batch_size", batch_size)

    # -- Data --
    print("=" * 60)
    print("COLORED MNIST COMPARISON EXPERIMENT")
    print("=" * 60)
    print(f"Methods: {method_list}")
    print(f"Baseline epochs: {baseline_epochs}, UNCP epochs: {uncp_epochs}")
    print(f"NSA samples: {nsa_samples}")
    print(f"Device: {device}")
    if args.quick:
        print(f"** QUICK MODE ** - subset={train_subset}, reduced epochs")
    print()

    print("[Data] Loading Colored MNIST (rho=0.9)...")
    train_loader, val_loader, id_test_loader, ood_test_loader = get_colored_mnist_dataloaders(config)
    print("[Data] Using anti-correlated test loader (rho=0.0) from repo dataloader.")

    # Subsample training data in quick mode
    if train_subset is not None and train_subset < len(train_loader.dataset):
        print(f"[Data] Subsampling to {train_subset} training examples...")
        subset_indices = np.random.choice(
            len(train_loader.dataset), train_subset, replace=False
        )
        train_subset_dataset = torch.utils.data.Subset(
            train_loader.dataset, subset_indices
        )
        train_loader = torch.utils.data.DataLoader(
            train_subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    # -- Run methods --
    all_results = []

    for method_name in method_list:
        print("\n" + "-" * 60)
        print(f"METHOD: {method_name.upper()}")
        print("-" * 60)

        if method_name == "uncp":
            # Handle UNCP separately
            if args.reuse_existing_uncp:
                print(f"[UNCP] Reusing existing results from {args.reuse_existing_uncp}")
                try:
                    with open(
                        os.path.join(args.reuse_existing_uncp, "results.json"), "r"
                    ) as f:
                        uncp_result = json.load(f)
                    all_results.append(uncp_result)
                    print(
                        "  Loaded: WGA="
                        f"{uncp_result.get('ood_worst_group_accuracy', uncp_result.get('worst_group_accuracy', 'N/A'))}"
                    )
                except Exception as e:
                    print(f"  Failed to load: {e}. Running UNCP from scratch.")
                    uncp_result = _run_uncp(
                        train_loader, val_loader, id_test_loader,
                        ood_test_loader, config, device,
                        uncp_epochs, nsa_samples,
                    )
                    all_results.append(uncp_result)
            else:
                uncp_result = _run_uncp(
                    train_loader, val_loader, id_test_loader,
                    ood_test_loader, config, device,
                    uncp_epochs, nsa_samples,
                )
                all_results.append(uncp_result)
            continue

        # Standard baseline
        try:
            model = create_resnet18_colored_mnist()
            trainer = get_baseline(
                name=method_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=ood_test_loader,
                config=config,
                device=device,
            )
            result = trainer.run()
            result = normalize_baseline_result(
                method_name, result, trainer,
                id_test_loader, ood_test_loader, device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [{method_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "method": method_name,
                "error": str(e),
                "ood_worst_group_accuracy": 0,
                "id_average_accuracy": 0,
            })

    # -- Compute SRD for all methods --
    print("\n" + "=" * 60)
    print("COMPUTING SRD FOR ALL METHODS")
    print("=" * 60)

    for i, result in enumerate(all_results):
        if result.get("error"):
            continue
        if result.get("srd") is not None:
            continue
        method_name = result["method"]
        print(f"  [{method_name}] Computing SRD...")

        # Re-create the model and load best weights
        # (SRD needs the actual model, not just metrics)
        # For baselines, the trainer's model is still in memory
        # We'll compute SRD from per-group accuracies instead
        try:
            per_group_id = result.get("id_per_group_accuracy", result.get("per_group_accuracy", {}))
            per_group_ood = result.get("ood_per_group_accuracy", result.get("per_group_accuracy", {}))

            # Compute SRD from per-group accuracies
            if per_group_id and per_group_ood:
                # Use the OOD per-group acc as "corrupted" and ID as "clean"
                # For Colored MNIST, the OOD set IS the corruption (removed color signal)
                degradations = {}
                for gi in per_group_id:
                    id_acc = per_group_id.get(gi, 0)
                    ood_acc = per_group_ood.get(gi, 0)
                    degradations[gi] = id_acc - ood_acc

                if degradations:
                    worst_g = max(degradations, key=degradations.get)
                    best_g = min(degradations, key=degradations.get)
                    srd = degradations[worst_g] - degradations[best_g]
                    srd_v2 = float(np.var(list(degradations.values())))
                else:
                    srd = None
                    srd_v2 = None

                result["srd"] = srd
                result["srd_v2"] = srd_v2
                result["per_group_degradation"] = degradations
        except Exception as e:
            print(f"    SRD computation failed: {e}")

    # -- Save results --
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / f"comparison_colored_mnist_{timestamp}"
    ensure_dir(str(output_dir))

    # JSON results
    save_json(all_results, str(output_dir / "results.json"))

    # LaTeX tables
    latex_main = format_main_results_table(all_results)
    with open(output_dir / "table_main_results.tex", "w") as f:
        f.write(latex_main)

    latex_comp = format_method_comparison_table(all_results)
    with open(output_dir / "table_method_comparison.tex", "w") as f:
        f.write(latex_comp)

    # Plots
    try:
        plot_wga_comparison(all_results, str(output_dir / "wga_comparison.png"))
        plot_srd_comparison(all_results, str(output_dir / "srd_comparison.png"))
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    # -- Print summary --
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Terminal table
    header = f"{'Method':<20} {'ID Avg':>8} {'OOD WGA':>8} {'SRD':>8} {'GrpLbl':>7} {'Cost':>6}"
    print(header)
    print("-" * len(header))

    for r in sorted(
        all_results,
        key=lambda r: r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0)),
        reverse=True,
    ):
        method = r.get("method", "unknown")
        id_avg = r.get("id_average_accuracy", r.get("average_accuracy", 0))
        ood_wga = r.get("ood_worst_group_accuracy", r.get("worst_group_accuracy", 0))
        srd = r.get("srd", None)
        gl = "Yes" if r.get("requires_group_labels") else "No"
        cost = r.get("relative_training_cost", "1x")

        srd_str = f"{srd:.3f}" if srd is not None else "---"
        print(f"{method:<20} {id_avg:>8.4f} {ood_wga:>8.4f} {srd_str:>8} {gl:>7} {cost:>6}")

    print()
    print(f"Results saved to: {output_dir}")
    print(f"  results.json")
    print(f"  table_main_results.tex")
    print(f"  table_method_comparison.tex")
    print(f"  wga_comparison.png")
    print(f"  srd_comparison.png")


def _run_uncp(
    train_loader, val_loader, id_test_loader,
    ood_test_loader, config, device,
    uncp_epochs, nsa_samples,
):
    """Wrapper to run UNCP pipeline and time it."""
    start = time.time()
    result, nsp_before, nsp_after, model = run_uncp_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        id_test_loader=id_test_loader,
        ood_test_loader=ood_test_loader,
        config=config,
        device=device,
        nsa_samples=nsa_samples,
        uncp_epochs=uncp_epochs,
    )
    result["training_time_seconds"] = time.time() - start

    # Compute SRD for UNCP model
    try:
        srd_results = compute_srd_for_model(
            model, id_test_loader, ood_test_loader, device
        )
        result["srd"] = srd_results.get("srd")
        result["srd_v2"] = srd_results.get("srd_v2")
    except Exception as e:
        print(f"  [UNCP SRD] Failed: {e}")

    return result


if __name__ == "__main__":
    main()
