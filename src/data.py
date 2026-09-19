"""Fashion-MNIST 数据加载。

原始图像为 28x28 灰度图。为与 LeNet-5 论文中 32x32 输入对齐，
使用 2 像素零填充，并标准化到均值 0.5、标准差 0.5。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _worker_init_fn(worker_id: int) -> None:
    """让每个 DataLoader worker 的 numpy / random 与 torch seed 对齐。"""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Pad(2),  # 28x28 -> 32x32，对齐原始 LeNet-5
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_raw_transform() -> transforms.Compose:
    """仅转 Tensor，用于展示原始样例。"""
    return transforms.Compose([transforms.ToTensor()])


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, datasets.FashionMNIST]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_full = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=get_transforms(),
    )
    test_set = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=get_transforms(),
    )
    raw_train = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=get_raw_transform(),
    )

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    shuffle_gen = torch.Generator().manual_seed(seed)
    workers = max(0, int(num_workers))
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_worker_init_fn if workers > 0 else None,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, generator=shuffle_gen, **loader_kwargs
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, raw_train
