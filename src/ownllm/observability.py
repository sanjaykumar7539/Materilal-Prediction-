"""Lightweight process metrics."""
from __future__ import annotations

import threading
import time


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int | float] = {}
        self.started = time.time()

    def inc(self, key: str, n: int | float = 1) -> None:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def set(self, key: str, value: int | float) -> None:
        with self._lock:
            self.counters[key] = value

    def snapshot(self) -> dict:
        with self._lock:
            return {"uptime_s": round(time.time() - self.started, 2), **dict(self.counters)}


metrics = Metrics()
