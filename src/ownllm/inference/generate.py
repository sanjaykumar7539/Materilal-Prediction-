"""Autoregressive sampling utilities."""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_ids: Sequence[int],
    penalty: float = 1.0,
) -> torch.Tensor:
    """Down-weight logits of already-generated tokens (presence penalty).

    logits: [B, V]; seen_ids: token ids to penalize. penalty <= 1.0 is a no-op.
    """
    if penalty <= 1.0 or not seen_ids:
        return logits
    out = logits.clone()
    idx = torch.tensor(sorted(set(int(i) for i in seen_ids)), device=logits.device)
    out[:, idx] = out[:, idx] / penalty
    return out


def sample(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Sample next token ids from logits [B, V] -> [B, 1]."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    x = logits / max(temperature, 1e-8)
    if top_k and top_k > 0:
        k = min(top_k, x.size(-1))
        values, _ = torch.topk(x, k)
        threshold = values[..., -1, None]
        x = x.masked_fill(x < threshold, float("-inf"))

    probs = torch.softmax(x, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def token_stream(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new: int = 128,
    eos: Optional[int] = None,
    repetition_penalty: float = 1.0,
    **sampling,
) -> Iterator[torch.Tensor]:
    """Yield next-token tensors with KV cache (never yields the EOS id)."""
    caches = None
    ids = prompt
    logits, _, caches = model(ids, caches=caches)
    seen: list[int] = ids.reshape(-1).tolist()
    nxt = sample(
        apply_repetition_penalty(logits[:, -1], seen, repetition_penalty), **sampling
    )
    for _ in range(max_new):
        if eos is not None and bool((nxt == eos).all()):
            break
        yield nxt
        seen.append(int(nxt.reshape(-1)[0]))
        logits, _, caches = model(nxt, caches=caches)
        nxt = sample(
            apply_repetition_penalty(logits[:, -1], seen, repetition_penalty), **sampling
        )
