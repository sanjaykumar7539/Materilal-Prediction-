"""Preference / alignment losses (DPO)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels.ne(-100)
    safe = labels.masked_fill(~mask, 0)
    token_lp = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi = policy_chosen - policy_rejected
    ref = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi - ref)).mean()
