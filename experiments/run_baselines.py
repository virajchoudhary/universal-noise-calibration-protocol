"""Phase 1 experiment: ERM baseline on Colored MNIST.

EXPECTED: ERM should achieve high average accuracy (~95%) on the in-distribution
test set (ρ=0.9) but very low worst-group / anti-correlated accuracy (20-30%),
because the network relies on color as a shortcut. This failure is the
MOTIVATION for UNCP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.erm import ERMTrainer
from uncp.data.colored_mnist import get_colored_mnist_dataloaders
from uncp.models import build_model
from uncp.utils import get_device, save_json, set_seed, timestamped_dir


def main(config_path: str = "configs/vision/colored_mnist.yaml",
         epochs_override: int | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / config_path)
    if epochs_override is not None:
        cfg.training.epochs = int(epochs_override)

    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))
    print(f"[setup] device={device}  seed={cfg.seed}  epochs={cfg.training.epochs}")

    train_loader, val_loader, test_loader, test_shift = get_colored_mnist_dataloaders(cfg)
    print(f"[data] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  shifted={len(test_shift.dataset)}")

    model = build_model(cfg.model.name, num_classes=int(cfg.model.num_classes))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model.name}  params={n_params/1e6:.2f}M")

    trainer = ERMTrainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        shifted_test_loader=test_shift,
        config=cfg, device=device, run_name="erm_colored_mnist",
        use_wandb=bool(cfg.wandb.enabled),
        checkpoint_dir=str(root / "checkpoints"),
    )
    results = trainer.run()

    out = timestamped_dir(root / "results", prefix="erm_colored_mnist")
    save_json({**results, "config": OmegaConf.to_container(cfg, resolve=True)},
              out / "results.json")

    test = results["test"]; shifted = results["shifted_test"]
    print()
    print("=" * 60)
    print("ERM on Colored MNIST (ρ=0.9) — Phase 1 result")
    print("=" * 60)
    print(f"  Avg acc (id,   ρ=0.9):  {test['acc']*100:6.2f}%")
    print(f"  Avg acc (ood,  ρ=0.0):  {shifted['acc']*100:6.2f}%")
    print(f"  WGA on id test:         {(test['worst_group_acc'] or 0)*100:6.2f}%")
    print(f"  WGA on ood test:        {(shifted['worst_group_acc'] or 0)*100:6.2f}%")
    print(f"  Per-group (id):         {test['per_group_acc']}")
    print(f"  Per-group (ood):        {shifted['per_group_acc']}")
    print(f"  Saved → {out}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--config", default="configs/vision/colored_mnist.yaml")
    args = p.parse_args()
    main(args.config, args.epochs)
