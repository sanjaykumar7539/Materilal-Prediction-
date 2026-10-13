#!/usr/bin/env python3
"""Generic local embedding trainer skeleton.
Input batches must provide positive text pairs tokenized with the OWN tokenizer.
"""
from __future__ import annotations

import torch

from ownllm.embeddings import LocalEmbeddingEncoder, contrastive_loss


def train_batch(
    model: LocalEmbeddingEncoder,
    opt: torch.optim.Optimizer,
    a_ids: torch.Tensor,
    b_ids: torch.Tensor,
) -> float:
    model.train()
    opt.zero_grad()
    loss = contrastive_loss(model(a_ids), model(b_ids))
    loss.backward()
    opt.step()
    return float(loss)


if __name__ == "__main__":
    print("train_embeddings: import train_batch(model, opt, a_ids, b_ids) in your loop.")
