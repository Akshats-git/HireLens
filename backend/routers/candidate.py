"""
Candidate-facing endpoint.

POST /api/candidate/analyze
  - Accepts a resume PDF + job description text
  - Returns overall score, 4-component breakdown, missing skills, suggestions
  - Target latency: < 3 seconds (GPU path with warm models)
"""

import asyncio
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ..schemas.candidate import AnalyzeResponse
from ..schemas.common import ScoreBreakdown
from ..services.cache_service import get_cache
from ..services.ml_service import get_ml_service
from ..services.pdf_service import extract_text

router = APIRouter(prefix="/api/candidate", tags=["candidate"])

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
_ACCEPTED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a resume against a job description",
    responses={
        415: {"description": "Non-PDF file uploaded"},
        413: {"description": "File exceeds 10 MB"},
        422: {"description": "PDF could not be parsed or text is too short"},
    },
)
async def analyze_resume(
    resume: UploadFile = File(..., description="Candidate resume in PDF format"),
    job_description: str = Form(..., min_length=50, description="Full job description text"),
) -> AnalyzeResponse:
    t_start = time.perf_counter()

    # ── Validate upload ────────────────────────────────────────────────────────
    content_type = (resume.content_type or "").lower()
    filename = resume.filename or "resume.pdf"
    if content_type not in _ACCEPTED_CONTENT_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    file_bytes = await resume.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB size limit.")

    # ── Extract text from PDF ──────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        resume_text: str = await loop.run_in_executor(None, extract_text, file_bytes, filename)
    except Exception as exc:
        logger.error(f"PDF extraction error [{filename}]: {exc}")
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")

    if len(resume_text) < 100:
        raise HTTPException(
            status_code=422,
            detail="Could not extract sufficient text from the PDF. "
                   "Ensure the file is not scanned-only (image-based).",
        )

    # ── Cache lookup ───────────────────────────────────────────────────────────
    cache = get_cache()
    cache_key = cache.make_analysis_key(resume_text, job_description)
    cached = cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for {filename}")
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return _to_response(cached, elapsed_ms)

    # ── ML inference ───────────────────────────────────────────────────────────
    ml = get_ml_service()
    try:
        result: dict = await loop.run_in_executor(None, ml.analyze, resume_text, job_description)
    except Exception as exc:
        logger.error(f"ML analysis failed [{filename}]: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    cache.set(cache_key, result)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"analyze | {filename} | score={result['score_pct']:.1f} | {elapsed_ms:.0f}ms")
    return _to_response(result, elapsed_ms)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _to_response(result: dict, elapsed_ms: float) -> AnalyzeResponse:
    bd = result["breakdown"]
    return AnalyzeResponse(
        score=result["score_pct"],
        label=result["label"],
        breakdown=ScoreBreakdown(
            skills_match=bd["skills_match"],
            experience_relevance=bd["experience_relevance"],
            education_fit=bd["education_fit"],
            keyword_alignment=bd["keyword_alignment"],
        ),
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        suggestions=result.get("suggestions", []),
        processing_time_ms=round(elapsed_ms, 1),
    )
