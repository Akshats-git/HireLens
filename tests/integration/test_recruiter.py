"""
Integration tests for recruiter endpoints:
  POST /api/recruiter/bulk-analyze
  GET  /api/recruiter/candidate/{id}
  POST /api/recruiter/filter
"""

import pytest

from tests.conftest import MINIMAL_PDF, MOCK_JD_TEXT, MOCK_RESULT
from backend.routers import recruiter as rec_module


# ── Helpers ────────────────────────────────────────────────────────────────────


def _upload_files(count: int = 2):
    return [
        ("resumes", (f"candidate_{i}.pdf", MINIMAL_PDF, "application/pdf"))
        for i in range(count)
    ]


# ── Bulk-analyze happy path ────────────────────────────────────────────────────


def test_bulk_analyze_returns_200(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(2),
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 200


def test_bulk_analyze_response_shape(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(2),
        data={"job_description": MOCK_JD_TEXT},
    )
    body = resp.json()
    assert "total" in body
    assert "candidates" in body
    assert "processing_time_ms" in body
    assert body["total"] == len(body["candidates"])


def test_bulk_analyze_candidate_shape(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(1),
        data={"job_description": MOCK_JD_TEXT},
    )
    c = resp.json()["candidates"][0]
    for field in ("id", "filename", "score", "label", "breakdown", "rank"):
        assert field in c, f"candidate missing field: {field}"
    assert c["rank"] == 1


def test_bulk_analyze_ranked_descending(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(3),
        data={"job_description": MOCK_JD_TEXT},
    )
    candidates = resp.json()["candidates"]
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    ranks = [c["rank"] for c in candidates]
    assert ranks == list(range(1, len(candidates) + 1))


def test_bulk_analyze_stores_candidates(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(2),
        data={"job_description": MOCK_JD_TEXT},
    )
    body = resp.json()
    for c in body["candidates"]:
        assert c["id"] in rec_module._candidate_store


# ── Error cases for bulk-analyze ──────────────────────────────────────────────


def test_bulk_analyze_no_files_returns_400(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code in (400, 422)


def test_bulk_analyze_short_jd_returns_422(client):
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(1),
        data={"job_description": "Too short"},
    )
    assert resp.status_code == 422


def test_bulk_analyze_oversized_file_returns_413(client):
    big_pdf = b"%PDF-1.4\n" + b"X" * (11 * 1024 * 1024)
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=[("resumes", ("big.pdf", big_pdf, "application/pdf"))],
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 413


# ── GET /candidate/{id} ────────────────────────────────────────────────────────


def test_get_candidate_returns_full_breakdown(client):
    # First create a candidate via bulk-analyze
    resp = client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(1),
        data={"job_description": MOCK_JD_TEXT},
    )
    cid = resp.json()["candidates"][0]["id"]

    resp2 = client.get(f"/api/recruiter/candidate/{cid}")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["id"] == cid
    assert "breakdown" in body
    assert "matched_skills" in body
    assert "suggestions" in body


def test_get_candidate_not_found_returns_404(client):
    resp = client.get("/api/recruiter/candidate/nonexistent-uuid")
    assert resp.status_code == 404


# ── POST /filter ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=False)
def seeded_store(client):
    """Ensure the store has at least some candidates before filter tests."""
    client.post(
        "/api/recruiter/bulk-analyze",
        files=_upload_files(3),
        data={"job_description": MOCK_JD_TEXT},
    )


def test_filter_no_constraints_returns_all(client, seeded_store):
    resp = client.post("/api/recruiter/filter", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert len(body["candidates"]) == body["total"]


def test_filter_min_score_above_all_returns_empty(client, seeded_store):
    resp = client.post("/api/recruiter/filter", json={"min_score": 100.0})
    assert resp.status_code == 200
    # Mock score is 78, so nothing passes a min_score of 100
    assert resp.json()["total"] == 0


def test_filter_min_score_below_all_passes_all(client, seeded_store):
    resp_all = client.post("/api/recruiter/filter", json={})
    resp_low = client.post("/api/recruiter/filter", json={"min_score": 0.0})
    assert resp_low.json()["total"] == resp_all.json()["total"]


def test_filter_must_have_matching_skill(client, seeded_store):
    # "python" is in mock matched_skills
    resp = client.post("/api/recruiter/filter", json={"must_have_skills": ["python"]})
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_filter_must_have_absent_skill_returns_empty(client, seeded_store):
    resp = client.post(
        "/api/recruiter/filter",
        json={"must_have_skills": ["this_skill_does_not_exist_xyz"]},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_filter_experience_level(client, seeded_store):
    # Mock sets experience_level = "mid"
    resp_mid = client.post("/api/recruiter/filter", json={"experience_level": "entry"})
    resp_lead = client.post("/api/recruiter/filter", json={"experience_level": "lead"})
    # entry level should include mid candidates; lead level should exclude them
    assert resp_mid.json()["total"] >= resp_lead.json()["total"]


def test_filter_results_ranked_descending(client, seeded_store):
    resp = client.post("/api/recruiter/filter", json={})
    candidates = resp.json()["candidates"]
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_filter_empty_store_returns_404(client):
    rec_module._candidate_store.clear()
    resp = client.post("/api/recruiter/filter", json={})
    assert resp.status_code == 404
