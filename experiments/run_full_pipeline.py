"""Phase 3 experiment: complete UNCP pipeline on Colored MNIST."""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncp.data.colored_mnist import get_colored_mnist_dataloaders
from uncp.training import UNCPPipelineVision
from uncp.utils import get_device, save_json, set_seed, timestamped_dir


def main(config_path: str = "configs/vision/colored_mnist.yaml",
         epochs_override: int | None = None,
         warm_start_epochs: int = 5) -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / config_path)
    if epochs_override is not None:
        cfg.training.epochs = int(epochs_override)
        cfg.training.phase_a_epochs = max(1, int(epochs_override * 0.15))
        cfg.training.phase_b_epochs = max(1, int(epochs_override * 0.60))
        cfg.training.phase_c_epochs = max(1,
            epochs_override - cfg.training.phase_a_epochs - cfg.training.phase_b_epochs)

    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))

    train, val, test, shifted = get_colored_mnist_dataloaders(cfg)
    run_dir = timestamped_dir(root / "results", prefix="uncp_colored_mnist")
    pipe = UNCPPipelineVision(
        config=cfg,
        train_loader=train, val_loader=val, test_loader=test,
        shifted_test_loader=shifted,
        device=device, run_dir=run_dir, run_name="uncp_colored_mnist",
    )
    results = pipe.run_full_pipeline(warm_start_epochs=warm_start_epochs)

    erm_id = results["erm_test_id"]
    erm_ood = results["erm_test_ood"]
    uncp_id = results["validation"]["test_id"]
    uncp_ood = results["validation"]["test_ood"]

    def fmt(v): return "    —" if v is None else f"{v*100:6.2f}"

    print()
    print("=" * 60)
    print("UNCP Results — Colored MNIST (ρ=0.9, η=0.25)")
    print("=" * 60)
    print(f"                           ERM     UNCP      Δ")
    print(f"Avg Acc  (id,  ρ=0.9)    {fmt(erm_id['acc'])}  {fmt(uncp_id['acc'])}  "
          f"{(uncp_id['acc']-erm_id['acc'])*100:+6.2f}")
    print(f"Avg Acc  (ood, ρ=0.0)    {fmt(erm_ood['acc'])}  {fmt(uncp_ood['acc'])}  "
          f"{(uncp_ood['acc']-erm_ood['acc'])*100:+6.2f}")
    wga_e = erm_ood['worst_group_acc'] or 0.0
    wga_u = uncp_ood['worst_group_acc'] or 0.0
    print(f"WGA      (ood, ρ=0.0)    {fmt(wga_e)}  {fmt(wga_u)}  "
          f"{(wga_u-wga_e)*100:+6.2f}")
    print(f"\nRecommended noise: {results['calibration']['recommended_noise_type']}")
    print(f"Calibrated σ: {results['calibration']['sigma_spurious']:.3f}")
    print(f"Phase B peak σ: {results['calibration']['sigma_high']:.3f}")
    print(f"\nSaved → {run_dir}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmstart", type=int, default=3)
    args = p.parse_args()
    main(epochs_override=args.epochs, warm_start_epochs=args.warmstart)
