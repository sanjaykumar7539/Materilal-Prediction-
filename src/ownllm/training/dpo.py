"""Direct Preference Optimization training step."""
from __future__ import annotations

import torch

from ownllm.alignment import dpo_loss, sequence_logp


def logp(model: torch.nn.Module, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, _, _ = model(ids)
    return sequence_logp(logits[:, :-1], labels[:, 1:])


def dpo_step(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    chosen_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pc = logp(policy, chosen_ids, chosen_labels)
    pr = logp(policy, rejected_ids, rejected_labels)
    with torch.no_grad():
        rc = logp(reference, chosen_ids, chosen_labels)
        rr = logp(reference, rejected_ids, rejected_labels)
    return dpo_loss(pc, pr, rc, rr, beta)
