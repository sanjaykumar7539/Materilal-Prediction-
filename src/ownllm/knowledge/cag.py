"""Cache-Augmented Generation helpers (local TTL + optional Redis)."""
from __future__ import annotations

import json
import time
from typing import Any, Optional


class LocalTTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._store[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache:
    def __init__(self, url: Optional[str] = None):
        self.url = url
        self.local = LocalTTLCache()
        self.redis = None

    async def connect(self) -> None:
        if not self.url:
            return
        try:
            import redis.asyncio as redis

            self.redis = redis.from_url(self.url, decode_responses=True)
            await self.redis.ping()
        except Exception:
            self.redis = None

    async def get(self, key: str) -> Optional[Any]:
        if self.redis is not None:
            try:
                raw = await self.redis.get(key)
                return None if raw is None else json.loads(raw)
            except Exception:
                pass
        return await self.local.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if self.redis is not None:
            try:
                await self.redis.set(key, json.dumps(value), ex=ttl)
                return
            except Exception:
                pass
        await self.local.set(key, value, ttl)
