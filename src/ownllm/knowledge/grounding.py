"""Evidence records and grounded context builder with citation map."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Evidence:
    id: str
    text: str
    source: Optional[str] = None
    score: float = 0.0


def build_grounded_context(
    evidence: list[Evidence],
    max_chars: int = 12000,
) -> tuple[str, dict]:
    used: list[Evidence] = []
    n = 0
    for e in evidence:
        text = e.text.strip()
        if not text:
            continue
        if n + len(text) > max_chars:
            break
        used.append(e)
        n += len(text)
    context = "\n\n".join(f"[{i + 1}] {e.text}" for i, e in enumerate(used))
    citations = {
        str(i + 1): {"id": e.id, "source": e.source, "score": e.score}
        for i, e in enumerate(used)
    }
    return context, citations
