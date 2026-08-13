"""
Integration tests for the recruiter endpoints:
  POST /api/recruiter/bulk-analyze
  POST /api/recruiter/filter
  GET  /api/recruiter/batches/{batch_id}/candidates/{candidate_id}
"""

import pytest

from backend.services.result_store import get_result_store
from tests.conftest import MINIMAL_PDF, MOCK_JD_TEXT


def _upload_files(count: int = 2):
    return [
        ("resumes", (f"candidate_{i}.pdf", MINIMAL_PDF, "application/pdf"))
        for i in range(count)
    ]


def _analyze(client, count: int = 3) -> dict:
    response = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(count),
        data={"job_description": MOCK_JD_TEXT},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def batch(client) -> dict:
    """A freshly analysed batch of three candidates."""
    return _analyze(client, count=3)


# ── Bulk analyze ───────────────────────────────────────────────────────────────


def test_bulk_analyze_returns_200(client):
    response = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(2),
        data={"job_description": MOCK_JD_TEXT},
    )
    assert response.status_code == 200


def test_bulk_analyze_response_shape(batch):
    for field in ("batch_id", "total", "submitted", "candidates", "processing_time_ms"):
        assert field in batch, f"response missing field: {field}"
    assert batch["total"] == len(batch["candidates"])
    assert batch["submitted"] == 3


def test_bulk_analyze_candidate_shape(batch):
    candidate = batch["candidates"][0]
    for field in ("id", "filename", "score", "label", "breakdown", "rank"):
        assert field in candidate, f"candidate missing field: {field}"
    assert candidate["rank"] == 1


def test_bulk_analyze_ranked_descending(batch):
    scores = [c["score"] for c in batch["candidates"]]
    assert scores == sorted(scores, reverse=True)
    ranks = [c["rank"] for c in batch["candidates"]]
    assert ranks == list(range(1, len(batch["candidates"]) + 1))


def test_bulk_analyze_stores_batch(batch):
    stored = get_result_store().get(batch["batch_id"])
    assert stored is not None
    assert {c["id"] for c in stored} == {c["id"] for c in batch["candidates"]}


def test_each_bulk_analyze_gets_its_own_batch(client):
    first = _analyze(client, count=2)
    second = _analyze(client, count=3)
    assert first["batch_id"] != second["batch_id"]


def test_filter_is_scoped_to_its_own_batch(client):
    """A batch must not leak candidates from an earlier upload."""
    first = _analyze(client, count=2)
    second = _analyze(client, count=3)

    response = client.post(
        "/api/recruiter/filter", json={"batch_id": second["batch_id"]}
    )
    assert response.status_code == 200

    returned = {c["id"] for c in response.json()["candidates"]}
    assert returned == {c["id"] for c in second["candidates"]}
    assert returned.isdisjoint({c["id"] for c in first["candidates"]})


# ── Bulk analyze error cases ───────────────────────────────────────────────────


def test_bulk_analyze_no_files_returns_400(client):
    response = client.post(
        "/api/recruiter/bulk-analyze", data={"job_description": MOCK_JD_TEXT}
    )
    assert response.status_code in (400, 422)


def test_bulk_analyze_short_jd_returns_422(client):
    response = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(1),
        data={"job_description": "Too short"},
    )
    assert response.status_code == 422


def test_bulk_analyze_oversized_file_returns_413(client):
    oversized = b"%PDF-1.4\n" + b"X" * (11 * 1024 * 1024)
    response = client.post(
        "/api/recruiter/bulk-analyze",
        files=[("resumes", ("big.pdf", oversized, "application/pdf"))],
        data={"job_description": MOCK_JD_TEXT},
    )
    assert response.status_code == 413


# ── Candidate lookup ───────────────────────────────────────────────────────────


def test_get_candidate_returns_full_breakdown(client, batch):
    candidate_id = batch["candidates"][0]["id"]
    response = client.get(
        f"/api/recruiter/batches/{batch['batch_id']}/candidates/{candidate_id}"
    )
    assert response.status_code == 200

    body = response.json()
    assert body["id"] == candidate_id
    for field in ("breakdown", "matched_skills", "suggestions"):
        assert field in body


def test_get_candidate_unknown_id_returns_404(client, batch):
    response = client.get(
        f"/api/recruiter/batches/{batch['batch_id']}/candidates/does-not-exist"
    )
    assert response.status_code == 404


def test_get_candidate_unknown_batch_returns_404(client, batch):
    candidate_id = batch["candidates"][0]["id"]
    response = client.get(
        f"/api/recruiter/batches/no-such-batch/candidates/{candidate_id}"
    )
    assert response.status_code == 404


# ── Filtering ──────────────────────────────────────────────────────────────────


def test_filter_no_constraints_returns_all(client, batch):
    response = client.post(
        "/api/recruiter/filter", json={"batch_id": batch["batch_id"]}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == len(batch["candidates"])
    assert len(body["candidates"]) == body["total"]


def test_filter_min_score_above_all_returns_empty(client, batch):
    response = client.post(
        "/api/recruiter/filter",
        json={"batch_id": batch["batch_id"], "min_score": 100.0},
    )
    assert response.json()["total"] == 0


def test_filter_min_score_below_all_passes_all(client, batch):
    response = client.post(
        "/api/recruiter/filter", json={"batch_id": batch["batch_id"], "min_score": 0.0}
    )
    assert response.json()["total"] == len(batch["candidates"])


def test_filter_must_have_matching_skill(client, batch):
    response = client.post(
        "/api/recruiter/filter",
        json={"batch_id": batch["batch_id"], "must_have_skills": ["python"]},
    )
    assert response.status_code == 200
    assert response.json()["total"] == len(batch["candidates"])


def test_filter_must_have_absent_skill_returns_empty(client, batch):
    response = client.post(
        "/api/recruiter/filter",
        json={
            "batch_id": batch["batch_id"],
            "must_have_skills": ["this_skill_does_not_exist_xyz"],
        },
    )
    assert response.json()["total"] == 0


def test_filter_experience_level_includes_equal_and_above(client, batch):
    """The mocked candidates are 'mid', so 'entry' includes them and 'lead' does not."""
    included = client.post(
        "/api/recruiter/filter",
        json={"batch_id": batch["batch_id"], "experience_level": "entry"},
    )
    excluded = client.post(
        "/api/recruiter/filter",
        json={"batch_id": batch["batch_id"], "experience_level": "lead"},
    )
    assert included.json()["total"] == len(batch["candidates"])
    assert excluded.json()["total"] == 0


def test_filter_rejects_unknown_experience_level(client, batch):
    response = client.post(
        "/api/recruiter/filter",
        json={"batch_id": batch["batch_id"], "experience_level": "principal"},
    )
    assert response.status_code == 422


def test_filter_results_ranked_descending(client, batch):
    response = client.post(
        "/api/recruiter/filter", json={"batch_id": batch["batch_id"]}
    )
    scores = [c["score"] for c in response.json()["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_filter_unknown_batch_returns_404(client):
    response = client.post("/api/recruiter/filter", json={"batch_id": "no-such-batch"})
    assert response.status_code == 404


def test_filter_requires_batch_id(client):
    response = client.post("/api/recruiter/filter", json={})
    assert response.status_code == 422
