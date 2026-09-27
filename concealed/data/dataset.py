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


def discover_images(root_dir: str | Path, deduplicate: bool = True) -> List[Path]:
    """Recursively discover all supported image files in ``root_dir``.

    When ``deduplicate=True`` (default), skips duplicate copies of the same filename
    and byte size in subdirectories (e.g., when a dataset folder contains both flat images
    and a ``by_category/`` subfolder copy).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [root]

    files: List[Path] = []
    seen_keys: set[tuple[str, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = Path(dirpath) / fname
                if deduplicate:
                    try:
                        key = (fname.lower(), full_path.stat().st_size)
                    except OSError:
                        key = (fname.lower(), -1)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                files.append(full_path)
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


def split_shared_val_and_train_shard(
    all_files: Sequence[Path],
    val_split: float = 0.1,
    seed: int = 42,
    max_images: Optional[int] = None,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Tuple[List[Path], List[Path]]:
    """Split files into a globally shared validation set and a worker-specific training shard.

    All shards receive the exact same ``val_files`` so validation analytics (PatchCos,
    Feature Re-ID Evasion, Semantic Flip) are 100% comparable across Snowflake accounts,
    while ``train_files`` are partitioned disjointly across ``num_shards``.
    """
    if not all_files:
        raise ValueError("No image files provided for splitting.")

    rng = random.Random(seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, shuffled

    # Carve out a globally consistent validation gallery first
    effective_for_val = min(len(shuffled), max_images) if (max_images and max_images > 0) else len(shuffled)
    val_count = max(1, int(round(effective_for_val * val_split)))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = shuffled[:val_count]
    train_pool = shuffled[val_count:]

    if num_shards > 1:
        if not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id must be in [0, {num_shards - 1}], got {shard_id}")
        train_files = train_pool[shard_id::num_shards]
    else:
        train_files = train_pool

    if max_images is not None and max_images > 0:
        target_train = max(1, max_images - len(val_files))
        if len(train_files) > target_train:
            train_files = train_files[:target_train]

    return train_files, val_files


def create_train_val_dataloaders(
    data_dir: str | Path,
    resolution: int = 512,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    max_images: Optional[int] = None,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Discover images in ``data_dir``, split into train/val sets, and return DataLoaders."""
    all_files = discover_images(data_dir)
    if not all_files:
        raise ValueError(f"No supported images found in {data_dir}")

    train_files, val_files = split_shared_val_and_train_shard(
        all_files=all_files,
        val_split=val_split,
        seed=seed,
        max_images=max_images,
        num_shards=num_shards,
        shard_id=shard_id,
    )

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

