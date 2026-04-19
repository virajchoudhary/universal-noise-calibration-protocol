"""Colored MNIST — the canonical spurious-correlation benchmark.

Each digit is colored with its class-specific color with probability
``correlation_strength``; otherwise a random color is chosen. The
``group_label`` is 0 for *correlated* examples (majority) and 1 for
*uncorrelated* (minority). Evaluation at ``correlation_strength=0.0`` (the
anti-correlated test set) isolates the model's reliance on color.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


CLASS_COLORS = np.array(
    [
        [1.0, 0.0, 0.0],   # 0 red
        [0.0, 0.0, 1.0],   # 1 blue
        [0.0, 1.0, 0.0],   # 2 green
        [1.0, 1.0, 0.0],   # 3 yellow
        [0.0, 1.0, 1.0],   # 4 cyan
        [1.0, 0.0, 1.0],   # 5 magenta
        [1.0, 0.5, 0.0],   # 6 orange
        [0.5, 0.0, 0.5],   # 7 purple
        [1.0, 0.75, 0.8],  # 8 pink
        [0.65, 0.16, 0.16],  # 9 brown
    ],
    dtype=np.float32,
)
CLASS_COLOR_NAMES = [
    "red", "blue", "green", "yellow", "cyan",
    "magenta", "orange", "purple", "pink", "brown",
]


class ColoredMNIST(Dataset):
    """Colored MNIST dataset with controllable spurious color-label correlation.

    Parameters
    ----------
    root : str
        Directory under which MNIST raw files are cached.
    correlation_strength : float
        Probability that a sample's color matches its digit class.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    download : bool
        Whether to download MNIST if missing.
    seed : int
        Seed controlling the color assignment RNG (for reproducibility).
    val_fraction : float
        Fraction of the MNIST train split held out for validation.
    """

    GROUP_NAMES = {0: "color_matches_digit", 1: "color_differs"}

    def __init__(
        self,
        root: str = "./data",
        correlation_strength: float = 0.9,
        split: str = "train",
        download: bool = True,
        seed: int = 42,
        val_fraction: float = 0.1,
        label_noise: float = 0.25,
    ) -> None:
        """See class docstring. ``label_noise`` injects uniform label flipping
        so that digit *shape* predicts the label only with accuracy
        ``1 - label_noise``. With ρ = 0.9 and label_noise = 0.25 the color
        shortcut is strictly more predictive than shape on the training
        distribution, inducing ERM to latch onto color — the failure mode UNCP
        aims to cure. Following Arjovsky et al. (2019, IRM)."""
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split}")
        if not 0.0 <= correlation_strength <= 1.0:
            raise ValueError("correlation_strength must be in [0, 1]")
        if not 0.0 <= label_noise < 1.0:
            raise ValueError("label_noise must be in [0, 1)")

        self.root = root
        self.split = split
        self.correlation_strength = float(correlation_strength)
        self.label_noise = float(label_noise)
        self._seed = seed

        base_train = split in ("train", "val")
        base = datasets.MNIST(root, train=base_train, download=download)

        images = base.data.numpy().astype(np.float32) / 255.0
        true_labels = base.targets.numpy().astype(np.int64)

        if base_train:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(true_labels))
            n_val = int(len(true_labels) * val_fraction)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            selected = train_idx if split == "train" else val_idx
            images = images[selected]
            true_labels = true_labels[selected]

        # Training labels: with prob label_noise replace with uniform random class.
        # The model receives these noisy labels during training; we keep the true
        # labels for evaluation (test/val splits always use true labels).
        rng_noise = np.random.RandomState(seed + 1000 + {"train": 1, "val": 2, "test": 3}[split])
        noise_mask = rng_noise.rand(len(true_labels)) < self.label_noise if split == "train" else np.zeros(len(true_labels), dtype=bool)
        random_labels = rng_noise.randint(0, 10, size=len(true_labels))
        observed_labels = np.where(noise_mask, random_labels, true_labels).astype(np.int64)

        # Colors are assigned based on the *observed* (possibly noisy) label —
        # so color correlates with the label the model actually sees during
        # training, while shape only correlates at rate (1 − label_noise).
        rng_color = np.random.RandomState(seed + {"train": 1, "val": 2, "test": 3}[split])
        match = rng_color.rand(len(true_labels)) < self.correlation_strength
        random_colors = rng_color.randint(0, 10, size=len(true_labels))
        color_labels = np.where(match, observed_labels, random_colors).astype(np.int64)

        # For evaluation purposes, the "task label" is always the TRUE label,
        # and group label = 0 if color matches true label (majority / aligned
        # with spurious cue), else 1.
        task_labels = true_labels
        group_labels = (color_labels != task_labels).astype(np.int64)

        colored = np.zeros((len(true_labels), 3, 28, 28), dtype=np.float32)
        for i in range(len(true_labels)):
            c = CLASS_COLORS[color_labels[i]]
            intensity = images[i]
            colored[i, 0] = intensity * c[0]
            colored[i, 1] = intensity * c[1]
            colored[i, 2] = intensity * c[2]

        self.images = torch.from_numpy(colored)
        # Training: the model trains against *observed* (possibly-noisy) labels,
        # which is what rewards the color shortcut. Val/test: we evaluate
        # against *true* labels — a model relying on color fails OOD.
        self.true_labels = torch.from_numpy(task_labels)
        self.observed_labels = torch.from_numpy(observed_labels)
        if split == "train":
            self.labels = self.observed_labels
        else:
            self.labels = self.true_labels
        self.color_labels = torch.from_numpy(color_labels)
        self.group_labels = torch.from_numpy(group_labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "color_label": self.color_labels[idx],
            "group_label": self.group_labels[idx],
        }

    def get_group_counts(self) -> Dict[int, int]:
        groups, counts = torch.unique(self.group_labels, return_counts=True)
        return {int(g.item()): int(c.item()) for g, c in zip(groups, counts)}

    def get_spurious_info(self) -> Dict[str, Any]:
        return {
            "correlation_strength": self.correlation_strength,
            "num_groups": 2,
            "group_names": self.GROUP_NAMES,
            "color_names": CLASS_COLOR_NAMES,
            "num_classes": 10,
            "num_colors": 10,
        }

    def create_synthetic_shift_test(self) -> "ColoredMNIST":
        """Return the anti-correlated evaluation split (ρ = 0)."""
        return ColoredMNIST(
            root=self.root,
            correlation_strength=0.0,
            split="test",
            download=False,
            seed=self._seed,
            label_noise=self.label_noise,
        )


def get_colored_mnist_dataloaders(
    config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test, test_shifted) dataloaders for Colored MNIST.

    ``test_shifted`` is the anti-correlated test set (ρ = 0) used for worst-group
    evaluation — this is the standard protocol in prior work.
    """
    root = config.dataset.root
    rho = float(config.dataset.correlation_strength)
    label_noise = float(config.dataset.get("label_noise", 0.0))
    seed = int(config.get("seed", 42))
    bs = int(config.training.batch_size)
    nw = int(config.training.get("num_workers", 0))

    train_ds = ColoredMNIST(root, correlation_strength=rho, split="train",
                            download=bool(config.dataset.download), seed=seed,
                            label_noise=label_noise)
    val_ds = ColoredMNIST(root, correlation_strength=rho, split="val",
                          download=False, seed=seed, label_noise=label_noise)
    test_ds = ColoredMNIST(root, correlation_strength=rho, split="test",
                           download=False, seed=seed, label_noise=label_noise)
    test_shift_ds = ColoredMNIST(root, correlation_strength=0.0, split="test",
                                 download=False, seed=seed, label_noise=label_noise)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_shift_loader = DataLoader(test_shift_ds, batch_size=bs, shuffle=False,
                                   num_workers=nw)
    return train_loader, val_loader, test_loader, test_shift_loader


if __name__ == "__main__":
    ds = ColoredMNIST("./data", correlation_strength=0.9, split="train",
                      download=True, seed=42)
    print(f"num samples: {len(ds)}  groups: {ds.get_group_counts()}")
    print(ds.get_spurious_info())

    try:
        import matplotlib.pyplot as plt

        Path("./results").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))
        for ax, i in zip(axes, range(5)):
            sample = ds[i]
            ax.imshow(sample["image"].permute(1, 2, 0).numpy())
            ax.set_title(f"y={sample['label'].item()} g={sample['group_label'].item()}",
                         fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("./results/colored_mnist_samples.png", dpi=150)
        print("Saved sample grid → results/colored_mnist_samples.png")
    except Exception as exc:
        print(f"[warn] could not save samples: {exc}")
