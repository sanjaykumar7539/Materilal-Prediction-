"""Supervised fine-tuning helpers (assistant-token masking)."""
from __future__ import annotations

import torch


def assistant_only_labels(
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[~assistant_mask.bool()] = -100
    return labels


def sft_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    labels = assistant_only_labels(input_ids, assistant_mask)
    _, loss, _ = model(input_ids, labels)
    return loss
