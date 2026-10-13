"""Trainable local pointwise reranker."""
from __future__ import annotations

import torch
from torch import nn


class LocalReranker(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def pool(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids).mean(dim=1)

    def forward(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        q = self.pool(query_ids)
        d = self.pool(doc_ids)
        return self.net(torch.cat([q, d], dim=-1)).squeeze(-1)
