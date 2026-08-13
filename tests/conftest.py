"""
Shared pytest fixtures.

ML models and PDF extraction are mocked so tests run without GPU/model files.
"""

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# ── Mock analysis result ───────────────────────────────────────────────────────

MOCK_RESULT = {
    "final_score": 0.78,
    "score_pct": 78.0,
    "label": "Good",
    "breakdown": {
        "skills_match": 0.82,
        "experience_relevance": 0.75,
        "education_fit": 0.70,
        "keyword_alignment": 0.65,
    },
    "matched_skills": ["python", "fastapi", "sql", "docker"],
    "missing_skills": ["kubernetes", "terraform"],
    "suggestions": [
        "Add Kubernetes experience to strengthen your infrastructure skill set.",
        "Mention any CI/CD pipeline work explicitly.",
    ],
    "experience_level": "mid",
}

MOCK_RESUME_TEXT = (
    "John Doe | john@example.com\n"
    "Software Engineer with 3 years of experience in Python, FastAPI, SQL, and Docker.\n"
    "Education: B.S. Computer Science, State University, 2020.\n"
    "Skills: Python, FastAPI, PostgreSQL, Docker, REST APIs, Git.\n"
    "Experience: Backend Engineer at Acme Corp (2021–present).\n"
    "Built microservices handling 10k req/s. Led a team of 3 engineers."
)

MOCK_JD_TEXT = (
    "We are looking for a Backend Engineer with strong Python and FastAPI skills. "
    "Experience with SQL databases and Docker is required. "
    "Knowledge of Kubernetes and Terraform is a plus. "
    "3+ years of professional software development experience expected."
)


# ── Minimal valid PDF ──────────────────────────────────────────────────────────
# Created with reportlab offline and stored as raw bytes so tests have no deps.


def _make_minimal_pdf() -> bytes:
    """Return a minimal syntactically-valid PDF with one text page."""
    content = (
        b"BT /F1 12 Tf 72 720 Td " b"(Python FastAPI SQL Docker Engineer resume) Tj ET"
    )
    length = len(content)
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(length).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000400 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n480\n%%EOF"
    )


MINIMAL_PDF = _make_minimal_pdf()


# ── Mock ML service fixture ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def mock_ml():
    svc = MagicMock()
    svc.is_ready = True
    svc.analyze.return_value = MOCK_RESULT
    svc.startup.return_value = None
    return svc


# ── TestClient fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client(mock_ml):
    """
    FastAPI TestClient with ML and PDF extraction mocked out.

    The startup lifespan is allowed to run but get_ml_service() returns
    the lightweight mock so no models are loaded.
    """
    with (
        patch("backend.main.get_ml_service", return_value=mock_ml),
        patch("backend.routers.candidate.get_ml_service", return_value=mock_ml),
        patch("backend.routers.recruiter.get_ml_service", return_value=mock_ml),
        patch("backend.routers.candidate.extract_text", return_value=MOCK_RESUME_TEXT),
        patch("backend.routers.recruiter.extract_text", return_value=MOCK_RESUME_TEXT),
    ):
        from backend.main import app

        with TestClient(app, raise_server_exceptions=True) as test_client:
            yield test_client


@pytest.fixture(autouse=True)
def isolated_state():
    """
    Reset shared server state between tests.

    The result store and analysis cache are process-wide singletons, so without
    this a batch or cached score from one test leaks into the next.
    """
    from backend.services.cache_service import get_cache
    from backend.services.result_store import get_result_store

    get_result_store().clear()
    get_cache()._memory.clear()
    yield
