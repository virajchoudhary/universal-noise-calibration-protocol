"""Phase 2 experiment: run the NSA diagnostic on an ERM-trained model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncp.data.colored_mnist import get_colored_mnist_dataloaders
from uncp.models import build_model
from uncp.nsa import NSPVisualizer, SensitivityProbe, get_noise_generators
from uncp.utils import get_device, save_json, set_seed, timestamped_dir


def main(config_path: str = "configs/vision/colored_mnist.yaml",
         checkpoint: str = "checkpoints/erm_colored_mnist_best.pt") -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / config_path)
    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))

    _, _, test_loader, _ = get_colored_mnist_dataloaders(cfg)
    model = build_model(cfg.model.name, num_classes=int(cfg.model.num_classes))
    ckpt = torch.load(root / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[nsa] loaded {checkpoint}  device={device}")

    noise_types = list(cfg.nsa.noise_types)
    generators = get_noise_generators("vision", noise_types=noise_types)
    print(f"[nsa] generators: {list(generators.keys())}")

    probe = SensitivityProbe(
        model=model, noise_generators=generators,
        num_samples=int(cfg.nsa.num_samples),
        device=device, batch_size=int(cfg.training.batch_size),
        model_name="ERM_ColoredMNIST", domain="vision",
    )
    nsp = probe.probe(test_loader)

    out = timestamped_dir(root / "results", prefix="nsa_erm")
    nsp.save(out / "nsp.pkl")
    viz = NSPVisualizer()
    viz.create_diagnostic_report(nsp, out)

    ranking = nsp.get_vulnerability_ranking()
    disparity = nsp.get_group_disparity()
    most_noise, most_flip = nsp.get_most_sensitive_noise()
    best_noise = max(disparity.items(), key=lambda x: x[1])[0]
    sigma_low = nsp.get_magnitude_at_threshold(best_noise, 0.5, group=1)
    sigma_high = nsp.get_magnitude_at_threshold(best_noise, 0.5, group=0)

    print()
    print("=" * 60)
    print("NSA Diagnostic Summary")
    print("=" * 60)
    print(f"  most-sensitive noise (overall):      {most_noise}  (flip={most_flip:.3f})")
    print(f"  recommended noise for CNI:           {best_noise}")
    print(f"  min σ for minority-group flip ≥ 0.5: {sigma_low}")
    print(f"  min σ for majority-group flip ≥ 0.5: {sigma_high}")
    print("  vulnerability ranking:")
    for nt, r in ranking:
        print(f"    {nt:<20s} {r:.3f}   group-disparity {disparity[nt]:.2f}")
    print(f"  report → {out}")
    save_json({"ranking": ranking, "group_disparity": disparity,
               "recommended_noise": best_noise,
               "sigma_low": sigma_low, "sigma_high": sigma_high}, out / "summary.json")
    return {"nsp_path": str(out / "nsp.pkl"), "recommended": best_noise}


if __name__ == "__main__":
    main()
