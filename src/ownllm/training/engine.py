"""AMP-aware training step engine with grad accum and clipping."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_accum: int = 1,
        max_grad: float = 1.0,
        amp: str = "bf16",
    ):
        self.model = model
        self.opt = optimizer
        self.grad_accum = max(1, grad_accum)
        self.max_grad = max_grad
        self.step = 0
        self.amp = amp if torch.cuda.is_available() else "none"
        self.dtype = torch.bfloat16 if self.amp == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")

    def train_batch(self, microbatches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Run one optimizer step over one or more microbatches.

        The loss is normalized by the ACTUAL number of supplied microbatches.
        This makes gradient accumulation mathematically correct even for a short
        final accumulation group.
        """
        batches = list(microbatches)
        if not batches:
            raise ValueError("train_batch requires at least one microbatch")
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        denom = len(batches)
        for batch in batches:
            ids, tgt = batch[:2]
            ctx = (
                torch.autocast("cuda", dtype=self.dtype)
                if self.amp != "none"
                else nullcontext()
            )
            with ctx:
                if len(batch) == 3:
                    from .factual import weighted_token_loss
                    logits, _, _ = self.model(ids)
                    loss = weighted_token_loss(logits, tgt, batch[2])
                else:
                    _, loss, _ = self.model(ids, tgt)
            if loss is None:
                raise RuntimeError("model returned no training loss")
            total += float(loss.detach().item())
            self.scaler.scale(loss / denom).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        return total / denom
