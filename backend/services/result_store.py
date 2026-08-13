"""
In-memory store of bulk-analysis results.

Recruiter filtering runs server-side over an earlier upload, so ranked results
must outlive the request that produced them. They are grouped into batches:
one batch per bulk-analyze call, addressed by an opaque ID the client passes
back when filtering.

Batches are per-upload rather than global. A single shared pool would let a
filter request return candidates from an unrelated upload — including another
user's — and would grow without bound for the process lifetime. Entries here
expire on a TTL and the batch count is capped.

This is deliberately process-local: results are cheap to recompute and are not
worth a database. Behind multiple workers a client may land on a worker that
does not hold its batch, which surfaces as a 404 and a re-upload. Sharing them
across workers would mean moving this into Redis.
"""

import threading
import time
import uuid
from typing import Optional

from backend.config import RESULT_STORE_TTL_SECONDS

# Upper bound on retained batches, evicted oldest-first. Guards against a burst
# of uploads pinning memory before any of them reach their TTL.
MAX_BATCHES = 64


class ResultStore:
    """Thread-safe, TTL-bounded map of batch ID to ranked candidate results."""

    def __init__(
        self,
        ttl_seconds: int = RESULT_STORE_TTL_SECONDS,
        max_batches: int = MAX_BATCHES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_batches = max_batches
        self._batches: dict[str, tuple[list[dict], float]] = {}
        self._lock = threading.Lock()

    def create(self, candidates: list[dict]) -> str:
        """
        Store a ranked candidate list and return its batch ID.

        Args:
            candidates: Serialised CandidateResult dicts, ranked best-first.

        Returns:
            The new batch ID.
        """
        batch_id = str(uuid.uuid4())
        expires_at = time.time() + self._ttl

        with self._lock:
            self._evict_expired()
            while len(self._batches) >= self._max_batches:
                oldest = min(self._batches, key=lambda key: self._batches[key][1])
                del self._batches[oldest]
            self._batches[batch_id] = (candidates, expires_at)

        return batch_id

    def get(self, batch_id: str) -> Optional[list[dict]]:
        """Return a batch's candidates, or None if it is unknown or expired."""
        with self._lock:
            entry = self._batches.get(batch_id)
            if entry is None:
                return None
            candidates, expires_at = entry
            if time.time() > expires_at:
                del self._batches[batch_id]
                return None
            return candidates

    def get_candidate(self, batch_id: str, candidate_id: str) -> Optional[dict]:
        """Return one candidate from a batch, or None if either ID is unknown."""
        candidates = self.get(batch_id)
        if candidates is None:
            return None
        return next((c for c in candidates if c["id"] == candidate_id), None)

    def clear(self) -> None:
        """Drop every batch. Intended for tests."""
        with self._lock:
            self._batches.clear()

    def _evict_expired(self) -> None:
        """Remove expired batches. Caller must hold the lock."""
        now = time.time()
        for batch_id in [
            key for key, (_, expires_at) in self._batches.items() if now > expires_at
        ]:
            del self._batches[batch_id]


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: Optional[ResultStore] = None


def get_result_store() -> ResultStore:
    """Return the shared ResultStore instance."""
    global _store
    if _store is None:
        _store = ResultStore()
    return _store
