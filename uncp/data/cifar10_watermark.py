"""
CIFAR-10 with injected spurious watermark for UNCP research.

Creates a controllable spurious correlation by adding semi-transparent
colored watermarks to training images, where watermark color correlates
with class label at a configurable strength rho.
"""

import os
import copy
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

# 10 distinct watermark colors (one per class)
WATERMARK_COLORS = {
    0: (255, 0, 0),      # airplane -> red
    1: (0, 255, 0),      # automobile -> green
    2: (0, 0, 255),      # bird -> blue
    3: (255, 255, 0),    # cat -> yellow
    4: (0, 255, 255),    # deer -> cyan
    5: (255, 0, 255),    # dog -> magenta
    6: (255, 128, 0),    # frog -> orange
    7: (128, 0, 255),    # horse -> purple
    8: (255, 128, 128),  # ship -> pink
    9: (128, 64, 0),     # truck -> brown
}


class CIFAR10Watermark(Dataset):
    """CIFAR-10 with semi-transparent watermark creating spurious correlation.

    Parameters
    ----------
    root : str
        Root directory for CIFAR-10 data.
    rho : float
        Correlation strength between watermark color and class label.
        1.0 = watermark always matches class, 0.0 = watermark random.
    split : str
        'train', 'val', or 'test'. Uses standard CIFAR-10 train/test splits.
        Val is created from first 5000 training samples.
    watermark_alpha : float
        Transparency of watermark overlay (0=invisible, 1=opaque).
    watermark_size : int
        Size of watermark patch in pixels (CIFAR-10 is 32x32).
    download : bool
        Whether to download CIFAR-10 if not found.
    seed : int
        Random seed for reproducibility of watermark assignment.
    """

    def __init__(
        self,
        root: str = "./data",
        rho: float = 0.9,
        split: str = "train",
        watermark_alpha: float = 0.3,
        watermark_size: int = 8,
        download: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.rho = rho
        self.watermark_alpha = watermark_alpha
        self.watermark_size = watermark_size

        # Load base CIFAR-10
        is_train = split in ("train", "val")
        cifar = datasets.CIFAR10(
            root=root, train=is_train, download=download,
        )

        # Split
        if split == "val":
            # Use first 5000 of training as validation
            cifar.data = cifar.data[:5000]
            cifar.targets = cifar.targets[:5000]
        elif split == "train":
            # Use remaining for training
            cifar.data = cifar.data[5000:]
            cifar.targets = cifar.targets[5000:]

        self.images = cifar.data  # (N, 32, 32, 3) uint8
        self.labels = np.array(cifar.targets)  # (N,) int
        self.num_classes = 10

        # Assign watermarks
        rng = np.random.RandomState(seed)
        self.watermark_labels = np.zeros_like(self.labels)
        for i in range(len(self.labels)):
            if rng.random() < rho:
                self.watermark_labels[i] = self.labels[i]  # correlated
            else:
                # Random different class watermark
                options = list(range(self.num_classes))
                options.remove(self.labels[i])
                self.watermark_labels[i] = rng.choice(options)

        # Group labels: 0=correlated (majority), 1=uncorrelated (minority)
        self.group_labels = (self.watermark_labels != self.labels).astype(np.int64)

        # Pre-apply watermarks
        self._apply_watermarks()

        # Transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ])

        print(f"[CIFAR10-Watermark] {split}: {len(self)} samples, rho={rho}")
        self._print_stats()

    def _apply_watermarks(self):
        """Apply semi-transparent colored watermarks to images."""
        self.watermarked_images = self.images.copy()
        wm_size = self.watermark_size
        alpha = self.watermark_alpha

        for i in range(len(self.images)):
            wm_class = self.watermark_labels[i]
            color = WATERMARK_COLORS[wm_class]

            # Bottom-right corner watermark
            img = self.watermarked_images[i].astype(np.float32)
            region = img[-wm_size:, -wm_size:, :]

            # Alpha blend
            for c in range(3):
                region[:, :, c] = (
                    (1 - alpha) * region[:, :, c] + alpha * color[c]
                )

            img[-wm_size:, -wm_size:, :] = region
            self.watermarked_images[i] = img.astype(np.uint8)

    def _print_stats(self):
        """Print group statistics."""
        for g in range(2):
            count = (self.group_labels == g).sum()
            name = "correlated (maj)" if g == 0 else "uncorrelated (min)"
            print(f"  Group {g} ({name}): {count}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.watermarked_images[idx]
        img = Image.fromarray(img)
        img = self.transform(img)

        label = self.labels[idx]
        group = self.group_labels[idx]
        wm_label = self.watermark_labels[idx]

        return img, label, group, wm_label

    def get_group_counts(self) -> Dict[int, int]:
        counts = {}
        for g in range(2):
            counts[g] = int((self.group_labels == g).sum())
        return counts

    def get_spurious_info(self) -> Dict:
        return {
            "correlation_type": "watermark_color_x_class",
            "correlation_strength": self.rho,
            "num_groups": 2,
            "group_names": {0: "correlated", 1: "uncorrelated"},
            "minority_groups": [1],
            "majority_groups": [0],
        }


class CIFAR10WatermarkClean(Dataset):
    """CIFAR-10 test set WITHOUT watermarks for clean evaluation.

    Same images but no watermark applied. Used to measure
    whether the model learned the actual features vs the watermark.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "test",
        download: bool = True,
    ):
        super().__init__()
        cifar = datasets.CIFAR10(
            root=root, train=(split != "test"), download=download,
        )
        self.images = cifar.data
        self.labels = np.array(cifar.targets)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        return img, self.labels[idx], 0, 0  # dummy group/wm


def get_cifar10_watermark_dataloaders(config):
    """Create CIFAR-10 watermark dataloaders from config.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader, clean_test_loader)
    """
    root = config.dataset.root if hasattr(config, "dataset") and hasattr(config.dataset, "root") else "./data"
    batch_size = config.training.batch_size if hasattr(config, "training") and hasattr(config.training, "batch_size") else 128
    rho = config.dataset.correlation_strength if hasattr(config, "dataset") and hasattr(config.dataset, "correlation_strength") else 0.9

    train_set = CIFAR10Watermark(root=root, rho=rho, split="train")
    val_set = CIFAR10Watermark(root=root, rho=rho, split="val")
    test_set = CIFAR10Watermark(root=root, rho=rho, split="test")
    clean_test = CIFAR10WatermarkClean(root=root, split="test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    clean_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, clean_loader