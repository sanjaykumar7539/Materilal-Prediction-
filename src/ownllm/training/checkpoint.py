"""Atomic checkpoint save/load."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch


def atomic_save(state: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, p)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
    step: int = 0,
    extra: Optional[dict] = None,
    config: Optional[dict] = None,
) -> None:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if config is not None:
        state["config"] = config
    elif hasattr(model, "cfg"):
        state["config"] = model.cfg.to_dict() if hasattr(model.cfg, "to_dict") else dict(model.cfg.__dict__)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    atomic_save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
