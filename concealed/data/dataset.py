"""Dataset and dataloader utilities for training and evaluating the obfuscation generator."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def discover_images(root_dir: str | Path) -> List[Path]:
    """Recursively discover all supported image files in ``root_dir``."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [root]

    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(Path(dirpath) / fname)
    files.sort()
    return files


class ImageObfuscationDataset(Dataset):
    """Unlabeled image dataset for training and evaluating the Amortized Generator Network."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        resolution: int = 512,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.image_paths = list(image_paths)
        self.resolution = int(resolution)
        self.is_train = bool(is_train)

        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.resolution,
                        scale=(0.6, 1.0),
                        ratio=(0.8, 1.25),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (self.resolution, self.resolution),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        attempts = 0
        while attempts < 5:
            path = self.image_paths[(idx + attempts) % len(self.image_paths)]
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    tensor = self.transform(rgb)
                    return torch.clamp(tensor, 0.0, 1.0)
            except Exception:
                attempts += 1
        # Fallback procedural pattern if corrupted files encountered
        return torch.rand(3, self.resolution, self.resolution)


def create_train_val_dataloaders(
    data_dir: str | Path,
    resolution: int = 512,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Discover images in ``data_dir``, split into train/val sets, and return DataLoaders."""
    all_files = discover_images(data_dir)
    if not all_files:
        raise ValueError(f"No supported images found in {data_dir}")

    rng = random.Random(seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        train_files = shuffled
        val_files = shuffled
    else:
        val_count = max(1, int(round(len(shuffled) * val_split)))
        val_files = shuffled[:val_count]
        train_files = shuffled[val_count:]

    train_ds = ImageObfuscationDataset(train_files, resolution=resolution, is_train=True)
    val_ds = ImageObfuscationDataset(val_files, resolution=resolution, is_train=False)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
