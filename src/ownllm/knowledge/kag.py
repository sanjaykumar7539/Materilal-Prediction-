"""In-memory knowledge graph for multi-hop traversal (KAG)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g: dict[str, list[dict]] = defaultdict(list)

    def add(self, src: str, rel: str, dst: str, source: Optional[str] = None) -> None:
        self.g[src].append({"relation": rel, "target": dst, "source": source})

    def traverse(
        self,
        start: str,
        max_hops: int = 2,
        max_nodes: int = 64,
    ) -> list[dict[str, Any]]:
        queue: deque = deque([(start, 0)])
        seen = {start}
        out: list[dict] = []
        while queue and len(seen) <= max_nodes:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for edge in self.g.get(node, []):
                out.append({"source_node": node, **edge, "hop": depth + 1})
                target = edge["target"]
                if target not in seen:
                    seen.add(target)
                    queue.append((target, depth + 1))
        return out
