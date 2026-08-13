"""
Cache for completed analysis results.

Uses Redis when reachable and an in-process TTL map otherwise, so the API runs
with no infrastructure in development and shares a cache across workers in
production. Redis failures after startup degrade to a miss rather than an error:
a cache is an optimisation, and a dead cache should not fail requests.
"""

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

from loguru import logger

from backend.config import ANALYSIS_CACHE_TTL_SECONDS

KEY_PREFIX = "hl:"

# Entries evicted from the in-memory store once it reaches this size, oldest
# expiry first. Redis enforces its own limit via maxmemory.
MAX_MEMORY_ENTRIES = 2048


def _connect_redis(url: str):
    """Return a live Redis client, or None if one cannot be established."""
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception as exc:
        logger.info(f"Redis unavailable at {url} ({exc}) — using in-memory cache.")
        return None


class _MemoryStore:
    """Thread-safe TTL map used when Redis is not available."""

    def __init__(self, max_entries: int = MAX_MEMORY_ENTRIES) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._max_entries = max_entries
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
            self._evict_expired()
            while len(self._data) >= self._max_entries:
                soonest = min(self._data, key=lambda k: self._data[k][1])
                del self._data[soonest]
            self._data[key] = (value, time.time() + ttl)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_expired(self) -> None:
        """Remove expired entries. Caller must hold the lock."""
        now = time.time()
        for key in [k for k, (_, exp) in self._data.items() if now > exp]:
            del self._data[key]


class CacheService:
    """
    Analysis result cache, backed by Redis when available.

    Keys are SHA-256 digests of the full input text, so callers never construct
    keys themselves and cannot collide.
    """

    def __init__(
        self, redis_url: Optional[str] = None, default_ttl: Optional[int] = None
    ) -> None:
        self._ttl = default_ttl or ANALYSIS_CACHE_TTL_SECONDS
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = _connect_redis(url)
        self._memory = _MemoryStore()

        backend = "Redis" if self._redis else "in-memory"
        logger.info(f"CacheService using {backend} backend (TTL={self._ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for a key, or None on a miss."""
        try:
            raw = self._redis.get(key) if self._redis else self._memory.get(key)
        except Exception as exc:
            logger.warning(f"Cache read failed, treating as a miss: {exc}")
            return None

        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.warning(f"Discarding malformed cache entry: {exc}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache a JSON-serialisable value. Failures are logged, never raised."""
        serialized = json.dumps(value, default=str)
        try:
            if self._redis:
                self._redis.setex(key, ttl or self._ttl, serialized)
            else:
                self._memory.setex(key, ttl or self._ttl, serialized)
        except Exception as exc:
            logger.warning(f"Cache write failed: {exc}")

    @staticmethod
    def make_key(*parts: str) -> str:
        """Hash the given parts into a namespaced cache key."""
        digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
        return KEY_PREFIX + digest

    def make_analysis_key(self, resume_text: str, jd_text: str) -> str:
        """
        Key an analysis by its full inputs.

        Hashing a truncated prefix would let two resumes that share an opening
        section — a common template header, for instance — collide and serve one
        candidate the other's scores.
        """
        return self.make_key("analysis", resume_text, jd_text)


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Return the shared CacheService instance."""
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
