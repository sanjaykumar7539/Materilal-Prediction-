"""DDP / FSDP helpers."""
from __future__ import annotations

import os

import torch
from torch.distributed import destroy_process_group, init_process_group


def init_distributed() -> tuple[int, int]:
    world = int(os.getenv("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_process_group(backend=backend)
    local = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return world, local


def wrap_fsdp(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.distributed.is_initialized():
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
    return FSDP(model, device_id=device_id)


def cleanup() -> None:
    if torch.distributed.is_initialized():
        destroy_process_group()
