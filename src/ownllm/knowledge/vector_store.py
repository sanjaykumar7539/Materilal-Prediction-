"""Local numpy-backed vector store (mmap-friendly)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LocalVectorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.vec: np.ndarray | None = None
        self.docs: list[dict] = []

    def build(self, vectors: Any, docs: list[dict]) -> "LocalVectorStore":
        self.path.mkdir(parents=True, exist_ok=True)
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
        v = v / norms
        np.save(self.path / "vectors.npy", v)
        (self.path / "docs.json").write_text(json.dumps(docs, ensure_ascii=False))
        self.vec = v
        self.docs = docs
        return self

    def load(self) -> "LocalVectorStore":
        self.vec = np.load(self.path / "vectors.npy", mmap_mode="r")
        self.docs = json.loads((self.path / "docs.json").read_text())
        return self

    def search(self, query: Any, k: int = 10) -> list[dict]:
        if self.vec is None:
            self.load()
        assert self.vec is not None
        q = np.asarray(query, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self.vec @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [{**self.docs[int(i)], "score": float(scores[i])} for i in idx]
