"""STL-10 data loading utilities used throughout the project."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

IMAGE_SIZE = 96
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2603, 0.2566, 0.2713]


@dataclass
class DataConfig:
    dataset_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    context_crop_ratio: float = 0.6
    target_crop_ratio: float = 0.4
    drop_last: bool = True
    subset_size: Optional[int] = None


class DualCropTransform:
    """Produce context and target crops with shared augmentations."""

    def __init__(self, context_ratio: float, target_ratio: float) -> None:
        self.context_ratio = float(context_ratio)
        self.target_ratio = float(target_ratio)
        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
            ]
        )

    def __call__(self, img):  # type: ignore[override]
        context = self._build_view(img, self.context_ratio)
        target = self._build_view(img, self.target_ratio)
        return context, target

    def _build_view(self, img, ratio: float):
        width, height = img.size
        base = min(width, height)
        crop = max(1, min(base, int(round(base * max(1e-3, ratio)))))
        if crop >= height or crop >= width:
            top = 0
            left = 0
        else:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        view = F.resized_crop(img, top, left, crop, crop, size=(IMAGE_SIZE, IMAGE_SIZE))
        return self.augment(view)


class STL10PairDataset(Dataset):
    """STL-10 unlabeled dataset that returns dual crops and indices."""

    def __init__(self, config: DataConfig) -> None:
        self.base = datasets.STL10(root=config.dataset_root, split="unlabeled", download=True)
        self.transform = DualCropTransform(config.context_crop_ratio, config.target_crop_ratio)
        total = len(self.base)
        if config.subset_size is not None and config.subset_size < total:
            generator = torch.Generator().manual_seed(0)
            perm = torch.randperm(total, generator=generator)
            self.indices = perm[: config.subset_size].tolist()
        else:
            self.indices = list(range(total))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, _ = self.base[base_idx]
        context, target = self.transform(img)
        # Return the position inside the subset (0..len(indices)-1) so callers such as
        # the DifficultyBuffer can size their state using ``len(dataset)`` without
        # needing to know whether STL10 was subsetted.
        return context, target, idx


def _build_eval_transform(train: bool) -> transforms.Compose:
    ops = []
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ]
    )
    return transforms.Compose(ops)


def create_unlabeled_loader(config: DataConfig) -> Tuple[DataLoader, int]:
    """Create the unlabeled STL-10 loader used for pretraining."""

    dataset = STL10PairDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.num_workers > 0,
    )
    return loader, len(dataset)


def create_labeled_loader(
    *,
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool = False,
) -> DataLoader:
    """Create a labeled STL-10 loader for evaluation scripts."""

    transform = _build_eval_transform(train=split == "train")
    dataset = datasets.STL10(root=root, split=split, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
