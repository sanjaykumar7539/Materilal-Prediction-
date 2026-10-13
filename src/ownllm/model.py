"""OwnLLM: from-scratch decoder-only Transformer.

Architecture: RMSNorm, RoPE, GQA, SwiGLU, SDPA, optional gradient checkpointing.
Weight-tied embedding / LM head. No external pretrained weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stability under AMP
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (x_f * rms)).type_as(x)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0):
    """Rotary position embeddings. q/k: [B, H, T, D], positions: [T]."""
    dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    # positions: [T] -> [T, dim/2]
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = angles.cos()[None, None, :, :]  # [1, 1, T, dim/2]
    sin = angles.sin()[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.cfg = cfg
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if cache is None else cache[0].shape[2]
        positions = torch.arange(past_len, past_len + T, device=x.device)
        q, k = apply_rope(q, k, positions, self.cfg.rope_theta)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k.detach(), v.detach())

        # GQA: repeat KV heads to match query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal masking:
        #  - No cache, T > 1  -> plain causal (SDPA's is_causal is correct).
        #  - No cache, T == 1 -> no masking needed.
        #  - With cache, T > 1 -> chunked prefill: SDPA's is_causal would align the
        #    triangle to the top-left and mask out ALL past keys, so build an explicit
        #    mask that keeps every past key and applies causality within the new chunk.
        #  - With cache, T == 1 -> single-token decode, all past keys visible.
        attn_mask = None
        is_causal = False
        if cache is None:
            is_causal = T > 1
        elif T > 1:
            keep = torch.ones(T, past_len + T, dtype=torch.bool, device=q.device).tril(
                diagonal=past_len
            )
            attn_mask = torch.zeros(
                T, past_len + T, dtype=q.dtype, device=q.device
            ).masked_fill(~keep, float("-inf"))
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(self.norm1(x), cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class OwnLLM(nn.Module):
    """Decoder-only language model trained from scratch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> "OwnLLM":
        self.gradient_checkpointing = enabled
        return self

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[list] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list]:
        x = self.tok_emb(input_ids)
        new_caches: list = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if caches is None else caches[i]
            if self.training and self.gradient_checkpointing and layer_cache is None:
                from torch.utils.checkpoint import checkpoint

                def _run(inp: torch.Tensor, blk: Block = block) -> torch.Tensor:
                    out, _ = blk(inp, None)
                    return out

                x = checkpoint(_run, x, use_reentrant=False)
                new_caches.append(None)
            else:
                x, nc = block(x, layer_cache)
                new_caches.append(nc)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches
