"""Sliding-window rate limiting with Redis (async) or local fallback."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Optional


class LocalSlidingWindow:
    def __init__(self, limit: int = 30, window: int = 60):
        self.limit = limit
        self.window = window
        self.history: dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def allow(self, key: str) -> tuple[bool, int]:
        async with self.lock:
            now = time.time()
            q = self.history[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                retry = max(1, int(self.window - (now - q[0])))
                return False, retry
            q.append(now)
            return True, 0


class DistributedRateLimiter:
    """Prefer Redis when available; fall back to in-process sliding window."""

    def __init__(self, redis_client: Any = None, limit: int = 30, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window
        self.local = LocalSlidingWindow(limit, window)

    async def allow(self, key: str) -> tuple[bool, int]:
        if self.redis is None:
            return await self.local.allow(key)
        try:
            now = time.time()
            zkey = f"ownllm:rl:{key}"
            member = f"{now}:{id(asyncio.current_task())}"
            pipe = self.redis.pipeline(transaction=True)
            pipe.zremrangebyscore(zkey, 0, now - self.window)
            pipe.zadd(zkey, {member: now})
            pipe.zcard(zkey)
            pipe.expire(zkey, self.window + 1)
            _, _, count, _ = await pipe.execute()
            if count > self.limit:
                await self.redis.zrem(zkey, member)
                return False, self.window
            return True, 0
        except Exception:
            # Redis failure must not take the service down
            return await self.local.allow(key)
