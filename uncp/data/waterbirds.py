"""
Waterbirds dataset loader for UNCP spurious correlation research.

Waterbirds (Sagawa et al., 2020) contains bird images (landbirds and waterbirds)
on backgrounds (land and water), with a strong spurious correlation between
bird type and background.

Groups:
  0: landbird on land   (majority)
  1: landbird on water  (minority)
  2: waterbird on land  (minority)
  3: waterbird on water (majority)

The dataset requires manual download of the CUB-200-2011 dataset and
the Waterbirds metadata. If the data is not found, prints instructions.
"""

import os
import copy
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class WaterbirdsDataset(Dataset):
    """Waterbirds dataset with spurious correlation between bird type and background.

    Parameters
    ----------
    root : str
        Root directory containing the waterbirds data.
        Expected structure:
            root/
              CUB_200_2011/
                images/
                  001.Black_footed_Albatross/
                  ...
              waterbirds_metadata/
                waterbird_complete_npy_csv_file.csv
    split : str
        One of 'train', 'val', 'test'.
    transform : callable, optional
        Image transforms. If None, uses standard ImageNet preprocessing.
    download : bool
        If True, prints download instructions.
    """

    # Official split sizes from Sagawa et al.
    SPLIT_SIZES = {"train": 4795, "val": 1199, "test": 5794}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        download: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split

        if transform is not None:
            self.transform = transform
        else:
            # Standard ImageNet preprocessing for ResNet
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

        # Try to load metadata
        metadata_path = self._find_metadata()
        if metadata_path is None:
            print("=" * 60)
            print("WATERBIRDS DATASET NOT FOUND")
            print("=" * 60)
            print("To use Waterbirds, you need:")
            print("1. Download CUB-200-2011 from:")
            print("   https://www.vision.caltech.edu/datasets/cub_200_2011/")
            print("2. Download Waterbirds metadata from:")
            print("   https://nlp.stanford.edu/data/waterbirds_metadata.tar.gz")
            print(f"3. Place both under: {self.root}/")
            print()
            print("Expected structure:")
            print(f"  {self.root}/CUB_200_2011/images/")
            print(f"  {self.root}/waterbirds_metadata/")
            print("=" * 60)
            self._initialized = False
            self.samples = []
            return

        self._initialized = True
        self._load_data(metadata_path)

    def _find_metadata(self):
        """Find the Waterbirds metadata CSV file."""
        candidates = [
            self.root / "waterbirds_metadata" / "waterbird_complete_npy_csv_file.csv",
            self.root / "metadata" / "waterbird_complete_npy_csv_file.csv",
            self.root / "waterbird_complete_npy_csv_file.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_data(self, metadata_path):
        """Load data from metadata CSV."""
        import pandas as pd

        df = pd.read_csv(str(metadata_path))

        # Filter by split
        # The metadata uses: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        split_idx = split_map[self.split]
        df = df[df["split"] == split_idx].reset_index(drop=True)

        # Extract info
        # Columns typically: img_filename, y (label), place (background), split
        # y: 0=landbird, 1=waterbird
        # place: 0=land, 1=water
        self.img_filenames = df["img_filename"].values
        self.labels = df["y"].values.astype(np.int64)
        self.places = df["place"].values.astype(np.int64)

        # Compute group labels
        # group = y * 2 + place
        # 0: landbird+land (majority)
        # 1: landbird+water (minority)
        # 2: waterbird+land (minority)
        # 3: waterbird+water (majority)
        self.group_labels = self.labels * 2 + self.places

        # Image base path
        self.img_base = self.root / "CUB_200_2011"

        print(f"[Waterbirds] {self.split}: {len(self)} samples")
        self._print_group_stats()

    def _print_group_stats(self):
        """Print group distribution."""
        if not self._initialized:
            return
        group_names = {
            0: "landbird+land (maj)",
            1: "landbird+water (min)",
            2: "waterbird+land (min)",
            3: "waterbird+water (maj)",
        }
        for g in range(4):
            count = (self.group_labels == g).sum()
            print(f"  Group {g} ({group_names[g]}): {count}")

    def __len__(self):
        return len(self.samples) if not self._initialized else len(self.img_filenames)

    def __getitem__(self, idx):
        if not self._initialized:
            raise RuntimeError("Waterbirds dataset not initialized. Download data first.")

        # Load image
        img_path = self.img_base / self.img_filenames[idx]
        image = Image.open(str(img_path)).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        group = self.group_labels[idx]
        place = self.places[idx]

        return image, label, group, place

    def get_group_counts(self) -> Dict[int, int]:
        """Return count of samples per group."""
        if not self._initialized:
            return {}
        counts = {}
        for g in range(4):
            counts[g] = int((self.group_labels == g).sum())
        return counts

    def get_spurious_info(self) -> Dict:
        """Return spurious correlation metadata."""
        return {
            "correlation_type": "bird_type_x_background",
            "num_groups": 4,
            "group_names": {
                0: "landbird+land",
                1: "landbird+water",
                2: "waterbird+land",
                3: "waterbird+water",
            },
            "minority_groups": [1, 2],
            "majority_groups": [0, 3],
        }


class WaterbirdsFallsbackDataset(Dataset):
    """Fallback Waterbirds dataset that generates synthetic data
    for testing when the real dataset is not available.

    This creates a simple proxy: colored rectangles on backgrounds,
    mimicking the bird/background spurious correlation structure.
    """

    def __init__(
        self,
        root: str = "./data/waterbirds_fallback",
        split: str = "train",
        correlation_strength: float = 0.95,
        num_samples: int = None,
        transform=None,
    ):
        super().__init__()
        self.correlation = correlation_strength

        # Default sample counts matching real Waterbirds
        default_sizes = {"train": 4795, "val": 1199, "test": 5794}
        self.num_samples = num_samples or default_sizes.get(split, 1000)
        self.split = split

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # Generate synthetic data
        self._generate_synthetic()

    def _generate_synthetic(self):
        """Generate synthetic bird/background data."""
        np.random.seed(42 if self.split == "train" else 43)

        # Labels: 0=landbird, 1=waterbird (roughly 50/50)
        self.labels = np.random.randint(0, 2, size=self.num_samples)

        # Places: correlated with labels
        self.places = np.zeros(self.num_samples, dtype=np.int64)
        for i in range(self.num_samples):
            if np.random.random() < self.correlation:
                self.places[i] = self.labels[i]  # Same as label (correlated)
            else:
                self.places[i] = 1 - self.labels[i]  # Opposite (minority)

        # Group labels
        self.group_labels = self.labels * 2 + self.places

        # Generate simple synthetic images
        # landbird = warm rectangle, waterbird = cool rectangle
        # land bg = brown, water bg = blue
        self.images = []
        for i in range(self.num_samples):
            img = np.zeros((224, 224, 3), dtype=np.float32)
            # Background
            if self.places[i] == 0:  # land
                img[:, :, 0] = 0.6  # brown
                img[:, :, 1] = 0.4
                img[:, :, 2] = 0.2
            else:  # water
                img[:, :, 0] = 0.1  # blue
                img[:, :, 1] = 0.3
                img[:, :, 2] = 0.7
            # Bird (center rectangle)
            if self.labels[i] == 0:  # landbird
                img[70:154, 70:154, 0] = 0.9  # warm
                img[70:154, 70:154, 1] = 0.7
                img[70:154, 70:154, 2] = 0.3
            else:  # waterbird
                img[70:154, 70:154, 0] = 0.3  # cool
                img[70:154, 70:154, 1] = 0.5
                img[70:154, 70:154, 2] = 0.9
            self.images.append(img)

        print(f"[Waterbirds-Synthetic] {self.split}: {self.num_samples} samples, "
              f"correlation={self.correlation}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = self.transform(img)
        label = int(self.labels[idx])
        group = int(self.group_labels[idx])
        place = int(self.places[idx])
        return img, label, group, place

    def get_group_counts(self) -> Dict[int, int]:
        counts = {}
        for g in range(4):
            counts[g] = int((self.group_labels == g).sum())
        return counts

    def get_spurious_info(self) -> Dict:
        return {
            "correlation_type": "bird_type_x_background (synthetic)",
            "num_groups": 4,
            "group_names": {
                0: "landbird+land",
                1: "landbird+water",
                2: "waterbird+land",
                3: "waterbird+water",
            },
            "minority_groups": [1, 2],
            "majority_groups": [0, 3],
        }


def get_waterbirds_dataloaders(config, use_fallback=True):
    """Create Waterbirds dataloaders.

    Tries real dataset first, falls back to synthetic if not available.

    Parameters
    ----------
    config : DictConfig
        Configuration with dataset.root, dataset.batch_size, etc.
    use_fallback : bool
        If True, generates synthetic data when real data is unavailable.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader)
    """
    root = config.dataset.root if hasattr(config, "dataset") and hasattr(config.dataset, "root") else "./data/waterbirds"
    batch_size = config.training.batch_size if hasattr(config, "training") and hasattr(config.training, "batch_size") else 64

    # Try real dataset
    train_set = WaterbirdsDataset(root=root, split="train", download=True)

    if not train_set._initialized:
        if use_fallback:
            print("[Waterbirds] Real data not found. Using synthetic fallback.")
            train_set = WaterbirdsFallsbackDataset(
                root=root, split="train",
                correlation_strength=0.95,
            )
            val_set = WaterbirdsFallsbackDataset(
                root=root, split="val",
                correlation_strength=0.95,
            )
            test_set = WaterbirdsFallsbackDataset(
                root=root, split="test",
                correlation_strength=0.95,
            )
        else:
            raise FileNotFoundError(
                "Waterbirds dataset not found and use_fallback=False. "
                "Download the data or set use_fallback=True."
            )
    else:
        val_set = WaterbirdsDataset(root=root, split="val")
        test_set = WaterbirdsDataset(root=root, split="test")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return train_loader, val_loader, test_loader