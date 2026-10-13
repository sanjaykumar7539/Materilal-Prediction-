"""SQLite-backed persistent knowledge graph."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class PersistentKAG:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS edges("
            "src TEXT, rel TEXT, dst TEXT, source TEXT, "
            "UNIQUE(src, rel, dst, source))"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
        self.db.commit()

    def add(self, src: str, rel: str, dst: str, source: str = "") -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)",
            (src, rel, dst, source),
        )
        self.db.commit()

    def traverse(self, start: str, max_hops: int = 2, limit: int = 100) -> list[dict[str, Any]]:
        seen = {start}
        frontier = [start]
        out: list[dict] = []
        for hop in range(1, max_hops + 1):
            nxt: list[str] = []
            for node in frontier:
                rows = self.db.execute(
                    "SELECT rel, dst, source FROM edges WHERE src = ? LIMIT ?",
                    (node, limit),
                )
                for rel, dst, source in rows:
                    out.append(
                        {
                            "source_node": node,
                            "relation": rel,
                            "target": dst,
                            "source": source,
                            "hop": hop,
                        }
                    )
                    if dst not in seen:
                        seen.add(dst)
                        nxt.append(dst)
                    if len(out) >= limit:
                        return out
            frontier = nxt
            if not frontier:
                break
        return out

    def close(self) -> None:
        self.db.close()
