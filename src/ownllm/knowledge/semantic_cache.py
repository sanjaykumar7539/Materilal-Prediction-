"""Embedding-similarity semantic cache backed by SQLite."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


class SemanticCache:
    def __init__(self, path: str | Path, threshold: float = 0.96):
        self.threshold = threshold
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache("
            "id INTEGER PRIMARY KEY, vec TEXT, value TEXT, expires REAL, version TEXT)"
        )
        self.db.commit()

    def get(self, qvec: Any, version: str) -> Optional[Any]:
        q = np.asarray(qvec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        best: Optional[tuple[float, Any]] = None
        rows = self.db.execute(
            "SELECT id, vec, value, expires, version FROM cache WHERE expires > ?",
            (time.time(),),
        )
        for _id, v, val, _exp, ver in rows:
            if ver != version:
                continue
            x = np.asarray(json.loads(v), dtype=np.float32)
            score = float(q @ x / max(float(np.linalg.norm(x)), 1e-9))
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, json.loads(val))
        return None if best is None else best[1]

    def set(self, vec: Any, value: Any, ttl: float, version: str) -> None:
        self.purge_expired()
        self.db.execute(
            "INSERT INTO cache(vec, value, expires, version) VALUES (?, ?, ?, ?)",
            (
                json.dumps([float(x) for x in vec]),
                json.dumps(value),
                time.time() + ttl,
                version,
            ),
        )
        self.db.commit()

    def purge_expired(self) -> int:
        """Delete expired rows; returns number removed. Keeps the table bounded."""
        cur = self.db.execute("DELETE FROM cache WHERE expires <= ?", (time.time(),))
        self.db.commit()
        return int(cur.rowcount if cur.rowcount is not None else 0)

    def invalidate_version(self, version: str) -> None:
        self.db.execute("DELETE FROM cache WHERE version = ?", (version,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
