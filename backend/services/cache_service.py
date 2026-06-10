"""
Cache service for embedding results.

Tries Redis first; if Redis is unavailable falls back to a thread-safe
in-memory TTL cache so the API works without infrastructure changes.
"""

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional


# ── Redis probe ────────────────────────────────────────────────────────────────


def _build_redis_client(url: str):
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


# ── In-memory TTL store ────────────────────────────────────────────────────────


class _MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}  # key → (json_value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._data[key]
                return None
            return value

    def setex(self, key: str, ttl: int, value: str) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ttl)

    def evict_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [k for k, (_, exp) in self._data.items() if now > exp]
            for k in expired:
                del self._data[k]


# ── Public interface ───────────────────────────────────────────────────────────


class CacheService:
    """
    Unified cache layer: Redis when available, in-memory otherwise.

    Keys are SHA-256 hashes of the input content, so callers never need to
    manage key collisions.
    """

    def __init__(
        self, redis_url: Optional[str] = None, default_ttl: int = 3600
    ) -> None:
        self._ttl = default_ttl
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = _build_redis_client(url)
        self._memory = _MemoryStore()
        backend = "Redis" if self._redis else "in-memory"
        from loguru import logger

        logger.info(
            f"CacheService initialised with {backend} backend (TTL={default_ttl}s)"
        )

    # ── Low-level ops ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        raw = self._redis.get(key) if self._redis else self._memory.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._ttl
        serialized = json.dumps(value, default=str)
        if self._redis:
            self._redis.setex(key, ttl, serialized)
        else:
            self._memory.setex(key, ttl, serialized)

    # ── Key helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(*parts: str) -> str:
        combined = "|".join(parts)
        return "hl:" + hashlib.sha256(combined.encode()).hexdigest()

    def make_analysis_key(self, resume_text: str, jd_text: str) -> str:
        return self.make_key("analysis", resume_text[:1000], jd_text[:1000])


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
