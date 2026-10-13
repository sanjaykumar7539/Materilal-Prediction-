"""Reciprocal Rank Fusion and simple reranker interface."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[Any, float] = defaultdict(float)
    payload: dict[Any, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = item.get("id", id(item))
            scores[key] += 1.0 / (k + rank)
            payload[key] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [dict(payload[i], fusion_score=s) for i, s in ranked]


class Reranker:
    """Score-passthrough baseline; swap for LocalReranker or cross-encoder."""

    def score(self, query: str, item: dict) -> float:
        return float(item.get("score", item.get("fusion_score", 0.0)))

    def rerank(self, query: str, items: list[dict], limit: int = 8) -> list[dict]:
        return sorted(items, key=lambda x: self.score(query, x), reverse=True)[:limit]
