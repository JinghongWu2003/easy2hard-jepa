"""STL-10 dataset utilities for DynaJEPA."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import STL10


_IMAGE_SIZE = 96
_STL10_MEAN = (0.4467, 0.4398, 0.4066)
_STL10_STD = (0.2603, 0.2566, 0.2713)


@dataclass
class DataConfig:
    """Configuration for STL-10 data loaders."""

    dataset_root: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    context_crop_ratio: float
    target_crop_ratio: float
    drop_last: bool
    subset_size: Optional[int] = None


def _build_augmentation(min_scale: float) -> T.Compose:
    """Create the random augmentation pipeline used for JEPA views."""

    color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return T.Compose(
        [
            T.RandomResizedCrop(_IMAGE_SIZE, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(_STL10_MEAN, _STL10_STD),
        ]
    )


def _build_eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(_IMAGE_SIZE),
            T.CenterCrop(_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(_STL10_MEAN, _STL10_STD),
        ]
    )


class _STL10ViewDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset returning two augmented views and the sample index."""

    def __init__(
        self,
        root: str,
        splits: Sequence[str],
        context_transform: T.Compose,
        target_transform: T.Compose,
        subset_size: Optional[int] = None,
    ) -> None:
        self.context_transform = context_transform
        self.target_transform = target_transform
        self.datasets = [STL10(root=root, split=split, download=True) for split in splits]
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
        self.total_size = total if subset_size is None else min(total, subset_size)

    def __len__(self) -> int:  # type: ignore[override]
        return self.total_size

    def _resolve_index(self, index: int) -> Tuple[int, int]:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_cum
        return dataset_idx, sample_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:  # type: ignore[override]
        if index < 0 or index >= self.total_size:
            raise IndexError(index)
        dataset_idx, sample_idx = self._resolve_index(index)
        image, _ = self.datasets[dataset_idx][sample_idx]
        context = self.context_transform(image)
        target = self.target_transform(image)
        return context, target, index


class _STL10LabeledDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset returning a single augmented view, label, and index."""

    def __init__(self, root: str, split: str, transform: Optional[T.Compose] = None) -> None:
        self.dataset = STL10(root=root, split=split, download=True)
        self.transform = transform or _build_eval_transform()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:  # type: ignore[override]
        image, label = self.dataset[index]
        image = self.transform(image)
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        return image, label_tensor, index


def create_unlabeled_loader(config: DataConfig) -> Tuple[DataLoader, int]:
    """Create the STL-10 unlabeled data loader used for JEPA pre-training."""

    context_transform = _build_augmentation(max(0.0, min(1.0, config.context_crop_ratio)))
    target_transform = _build_augmentation(max(0.0, min(1.0, config.target_crop_ratio)))
    dataset = _STL10ViewDataset(
        root=config.dataset_root,
        splits=("train", "unlabeled"),
        context_transform=context_transform,
        target_transform=target_transform,
        subset_size=config.subset_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    return loader, len(dataset)


def create_labeled_loader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    """Create a labeled STL-10 data loader for evaluation or feature export."""

    dataset = _STL10LabeledDataset(root=root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


__all__ = ["DataConfig", "create_unlabeled_loader", "create_labeled_loader"]
