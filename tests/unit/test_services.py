"""Unit tests for the backend service layer: cache, result store, seniority tiers."""

import time

import pytest

from backend.services.cache_service import CacheService, _MemoryStore
from backend.services.ml_service import years_to_level
from backend.services.result_store import ResultStore


# ── In-memory TTL store ───────────────────────────────────────────────────────


def test_memory_store_round_trips_a_value():
    store = _MemoryStore()
    store.setex("k", 60, "value")
    assert store.get("k") == "value"


def test_memory_store_misses_on_an_unknown_key():
    assert _MemoryStore().get("absent") is None


def test_memory_store_expires_entries():
    store = _MemoryStore()
    store.setex("k", 0, "value")
    time.sleep(0.01)
    assert store.get("k") is None


def test_memory_store_evicts_once_full():
    """Regression: nothing ever called the eviction path, so the map grew forever."""
    store = _MemoryStore(max_entries=3)
    for i in range(10):
        store.setex(f"k{i}", 60, str(i))
    assert len(store._data) <= 3


# ── Cache keys ────────────────────────────────────────────────────────────────


@pytest.fixture
def cache(monkeypatch) -> CacheService:
    """A CacheService pinned to its in-memory backend."""
    monkeypatch.setattr(
        "backend.services.cache_service._connect_redis", lambda url: None
    )
    return CacheService()


def test_keys_are_namespaced_and_deterministic(cache):
    key = cache.make_key("analysis", "a", "b")
    assert key.startswith("hl:")
    assert key == cache.make_key("analysis", "a", "b")


def test_different_inputs_produce_different_keys(cache):
    assert cache.make_key("a") != cache.make_key("b")


def test_analysis_key_covers_the_whole_resume(cache):
    """
    Regression: only the first 1000 characters were hashed, so two resumes
    sharing a template header collided and one candidate was served the other's
    scores.
    """
    shared_header = "ACME RESUME TEMPLATE\n" * 60
    assert len(shared_header) > 1000

    first = cache.make_analysis_key(shared_header + "Python engineer.", "Job posting.")
    second = cache.make_analysis_key(shared_header + "Pastry chef.", "Job posting.")
    assert first != second


def test_analysis_key_varies_with_the_job_description(cache):
    resume = "Python engineer with 5 years of experience."
    assert cache.make_analysis_key(resume, "Posting A") != cache.make_analysis_key(
        resume, "Posting B"
    )


def test_cache_round_trips_json(cache):
    cache.set("hl:test", {"score": 78.0, "skills": ["python"]})
    assert cache.get("hl:test") == {"score": 78.0, "skills": ["python"]}


def test_cache_miss_returns_none(cache):
    assert cache.get("hl:never-written") is None


def test_backend_failure_degrades_to_a_miss(cache):
    """A dead cache must not turn into a failed request."""

    class Broken:
        def get(self, key):
            raise ConnectionError("redis is gone")

        def setex(self, key, ttl, value):
            raise ConnectionError("redis is gone")

    cache._redis = Broken()
    assert cache.get("hl:anything") is None
    cache.set("hl:anything", {"a": 1})  # must not raise


# ── Result store ──────────────────────────────────────────────────────────────


def _candidates(prefix: str, count: int = 3) -> list[dict]:
    return [{"id": f"{prefix}-{i}", "score": float(i)} for i in range(count)]


def test_store_returns_what_was_put_in():
    store = ResultStore()
    batch_id = store.create(_candidates("a"))
    assert store.get(batch_id) == _candidates("a")


def test_unknown_batch_returns_none():
    assert ResultStore().get("no-such-batch") is None


def test_each_batch_is_isolated():
    """
    Regression: results lived in one process-wide dict, so filtering a batch
    returned candidates from every earlier upload.
    """
    store = ResultStore()
    first = store.create(_candidates("first"))
    second = store.create(_candidates("second"))

    assert first != second
    assert {c["id"] for c in store.get(first)}.isdisjoint(
        {c["id"] for c in store.get(second)}
    )


def test_batches_expire():
    store = ResultStore(ttl_seconds=0)
    batch_id = store.create(_candidates("a"))
    time.sleep(0.01)
    assert store.get(batch_id) is None


def test_batch_count_is_capped():
    store = ResultStore(max_batches=3)
    ids = [store.create(_candidates(f"b{i}")) for i in range(10)]
    assert sum(store.get(batch_id) is not None for batch_id in ids) <= 3


def test_oldest_batch_is_evicted_first():
    store = ResultStore(max_batches=2)
    oldest = store.create(_candidates("oldest"))
    for i in range(3):
        store.create(_candidates(f"newer{i}"))
    assert store.get(oldest) is None


def test_candidate_lookup_is_scoped_to_its_batch():
    store = ResultStore()
    first = store.create(_candidates("first"))
    second = store.create(_candidates("second"))

    assert store.get_candidate(first, "first-0") is not None
    assert store.get_candidate(second, "first-0") is None


def test_candidate_lookup_returns_none_for_an_unknown_batch():
    assert ResultStore().get_candidate("missing", "any") is None


def test_clear_empties_the_store():
    store = ResultStore()
    batch_id = store.create(_candidates("a"))
    store.clear()
    assert store.get(batch_id) is None


# ── Seniority tiers ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "years, expected",
    [
        (0, "entry"),
        (1.9, "entry"),
        (2, "mid"),
        (4.9, "mid"),
        (5, "senior"),
        (9.9, "senior"),
        (10, "lead"),
        (30, "lead"),
    ],
)
def test_years_map_to_seniority_tiers(years, expected):
    assert years_to_level(years) == expected


def test_negative_years_fall_back_to_the_lowest_tier():
    assert years_to_level(-1) == "entry"
