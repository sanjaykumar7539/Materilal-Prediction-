"""Perplexity evaluation and promotion gate."""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch


@torch.inference_mode()
def perplexity(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    model.eval()
    total = 0.0
    n = 0
    for ids, tgt in batches:
        _, loss, _ = model(ids, tgt)
        total += float(loss.item())
        n += 1
    loss = total / max(n, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 20.0))}


def promotion_gate(
    metrics: dict,
    max_ppl: Optional[float] = None,
    required: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    ok = True
    reasons: list[str] = []
    if max_ppl is not None and metrics.get("perplexity", 1e9) > max_ppl:
        ok = False
        reasons.append("perplexity")
    for key, threshold in (required or {}).items():
        if metrics.get(key, float("-inf")) < threshold:
            ok = False
            reasons.append(key)
    return {"promote": ok, "failed": reasons}
