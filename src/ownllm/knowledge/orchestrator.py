"""RAG + KAG + CAG fusion orchestrator."""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from .fusion import Reranker, reciprocal_rank_fusion
from .grounding import Evidence, build_grounded_context


class KnowledgeOrchestrator:
    """Domain-neutral retrieval orchestration.

    Pipeline: CAG cache -> dense + BM25 -> RRF -> rerank -> KAG multi-hop
    -> grounded context + citations.
    """

    def __init__(
        self,
        cache: Any,
        dense: Any = None,
        sparse: Any = None,
        graph: Any = None,
        reranker: Optional[Reranker] = None,
    ):
        self.cache = cache
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.reranker = reranker or Reranker()

    @staticmethod
    def _key(query: str) -> str:
        return "ctx:" + hashlib.sha256(query.encode()).hexdigest()

    async def context(
        self,
        query: str,
        entities: tuple[str, ...] = (),
        limit: int = 8,
        ttl: int = 300,
        principal: str | None = None,
        roles: tuple[str, ...] = (),
    ) -> dict:
        import json
        def allowed(item):
            if principal is None:
                return True
            return item.get('public') is True or item.get('owner') == principal or bool(set(item.get('roles',())) & set(roles))
        key = self._key(json.dumps([query,entities,limit,principal,sorted(roles)],sort_keys=True))
        cached = await self.cache.get(key)
        # A secured query always re-reads the sources: cached ACLs can be revoked.
        if cached is not None and principal is None:
            return cached

        lists: list[list[dict]] = []
        if self.dense is not None:
            lists.append([item for item in await self.dense.search(query, limit * 2) if allowed(item)])
        if self.sparse is not None:
            lists.append([item for item in await self.sparse.search(query, limit * 2) if allowed(item)])

        fused = reciprocal_rank_fusion(lists) if lists else []
        ranked = self.reranker.rerank(query, fused, limit)

        graph_items: list[dict] = []
        if self.graph is not None:
            for entity in entities:
                if hasattr(self.graph, "traverse"):
                    graph_items.extend(self.graph.traverse(entity, max_hops=2))

        graph_items = [item for item in graph_items if allowed(item)]
        combined = reciprocal_rank_fusion([ranked,graph_items])
        ranked = [item for item in self.reranker.rerank(query,combined,limit) if allowed(item)]
        evidence = [
            Evidence(
                str(x.get("id", "")),
                x.get("text", ""),
                x.get("source"),
                float(x.get("score", x.get("fusion_score", 0.0))),
            )
            for x in ranked
            if x.get("text")
        ]
        context, citations = build_grounded_context(evidence)
        result = {"context": context, "citations": citations, "graph": graph_items}
        await self.cache.set(key, result, ttl)
        return result
