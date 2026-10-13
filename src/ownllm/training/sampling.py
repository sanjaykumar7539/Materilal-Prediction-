"""DataLoader factory with optional DistributedSampler."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler: Optional[DistributedSampler] = (
        DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return loader, sampler
