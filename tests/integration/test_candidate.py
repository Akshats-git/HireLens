"""
Integration tests for POST /api/candidate/analyze.

ML inference and PDF extraction are mocked (see tests/conftest.py).
"""

import io
import pytest

from tests.conftest import MINIMAL_PDF, MOCK_JD_TEXT, MOCK_RESULT


# ── Happy path ─────────────────────────────────────────────────────────────────


def test_analyze_returns_200(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 200


def test_analyze_response_shape(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    body = resp.json()
    assert "score" in body
    assert "label" in body
    assert "breakdown" in body
    assert "matched_skills" in body
    assert "missing_skills" in body
    assert "suggestions" in body
    assert "processing_time_ms" in body

    bd = body["breakdown"]
    for key in (
        "skills_match",
        "experience_relevance",
        "education_fit",
        "keyword_alignment",
    ):
        assert key in bd, f"breakdown missing key: {key}"


def test_analyze_score_range(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    body = resp.json()
    assert 0.0 <= body["score"] <= 100.0
    assert body["label"] == MOCK_RESULT["label"]


def test_analyze_matched_skills(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    body = resp.json()
    assert isinstance(body["matched_skills"], list)
    assert isinstance(body["missing_skills"], list)
    assert isinstance(body["suggestions"], list)


# ── Error cases ────────────────────────────────────────────────────────────────


def test_analyze_missing_jd_returns_422(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        # No job_description field
    )
    assert resp.status_code == 422


def test_analyze_jd_too_short_returns_422(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        data={"job_description": "Too short"},
    )
    assert resp.status_code == 422


def test_analyze_empty_file_returns_400(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("empty.pdf", b"", "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 400


def test_analyze_non_pdf_returns_415(client):
    resp = client.post(
        "/api/candidate/analyze",
        files={
            "resume": (
                "doc.docx",
                b"PK\x03\x04fake",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 415


def test_analyze_oversized_file_returns_413(client):
    big_bytes = b"%PDF-1.4\n" + b"X" * (11 * 1024 * 1024)
    resp = client.post(
        "/api/candidate/analyze",
        files={"resume": ("huge.pdf", big_bytes, "application/pdf")},
        data={"job_description": MOCK_JD_TEXT},
    )
    assert resp.status_code == 413


# ── Cache behaviour ────────────────────────────────────────────────────────────


def test_analyze_second_call_uses_cache(client, mock_ml):
    """Second identical request should hit cache → ml.analyze called exactly once."""
    mock_ml.analyze.reset_mock()

    for _ in range(2):
        resp = client.post(
            "/api/candidate/analyze",
            files={"resume": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
            data={"job_description": MOCK_JD_TEXT},
        )
        assert resp.status_code == 200

    assert (
        mock_ml.analyze.call_count <= 1
    ), "ml.analyze should be called at most once when both requests are identical"
