"""Trainable local embedding encoder (no external API)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LocalEmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 384, hidden: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.emb(ids)
        if mask is None:
            mask = ids.ne(0)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.proj(pooled), dim=-1)


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    logits = a @ b.T / temp
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
