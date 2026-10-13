"""Distributed checkpoint helpers using torch.distributed.checkpoint when available."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_distributed_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any = None,
    step: int = 0,
) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed.checkpoint as dcp

        state: dict[str, Any] = {
            "model": model.state_dict(),
            "step": torch.tensor(step),
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        dcp.save(state, checkpoint_id=str(p))
    else:
        from .checkpoint import atomic_save

        state = {"model": model.state_dict(), "step": step}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        atomic_save(state, p / "checkpoint.pt")


def load_distributed_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any = None,
) -> int:
    p = Path(path)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed.checkpoint as dcp

        state: dict[str, Any] = {
            "model": model.state_dict(),
            "step": torch.tensor(0),
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        dcp.load(state, checkpoint_id=str(p))
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        return int(state["step"])
    state = torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])
