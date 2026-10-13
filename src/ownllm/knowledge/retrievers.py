"""Sparse (BM25) and dense retrieval primitives."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Optional


class BM25Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.toks = [self._tok(d.get("text", "")) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / max(len(self.toks), 1)
        self.df: Counter = Counter()
        for tokens in self.toks:
            self.df.update(set(tokens))

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower())

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        q_tokens = self._tok(query)
        n = len(self.docs)
        out: list[dict] = []
        for doc, tokens in zip(self.docs, self.toks):
            tf = Counter(tokens)
            score = 0.0
            for w in q_tokens:
                df = self.df[w]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * len(tokens) / max(self.avgdl, 1))
                score += idf * f * (self.k1 + 1) / max(denom, 1e-9)
            if score > 0:
                out.append({**doc, "score": score})
        return sorted(out, key=lambda x: x["score"], reverse=True)[:limit]


class DenseRetriever:
    def __init__(self, docs: list[dict], embed_fn: Callable[[list[str]], Any]):
        self.docs = docs
        self.embed_fn = embed_fn
        self.matrix = None

    async def build(self) -> None:
        import torch

        texts = [d.get("text", "") for d in self.docs]
        emb = self.embed_fn(texts)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        self.matrix = torch.nn.functional.normalize(emb.float(), dim=-1)

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        import torch

        if self.matrix is None:
            await self.build()
        assert self.matrix is not None
        q = self.embed_fn([query])
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q)
        q = torch.nn.functional.normalize(q.float(), dim=-1)[0]
        scores = self.matrix @ q
        k = min(limit, len(self.docs))
        idx = torch.topk(scores, k).indices.tolist()
        return [{**self.docs[i], "score": float(scores[i])} for i in idx]
