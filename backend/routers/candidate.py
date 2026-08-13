"""
Candidate endpoint.

POST /api/candidate/analyze
    Score one resume PDF against a pasted job description and return the overall
    match, the four-component breakdown, missing skills, and suggestions.
"""

import asyncio
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ..config import (
    MAX_JD_CHARS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MIN_JD_CHARS,
    MIN_RESUME_CHARS,
)
from ..schemas.candidate import AnalyzeResponse
from ..schemas.common import ScoreBreakdown
from ..services.cache_service import get_cache
from ..services.ml_service import get_ml_service
from ..services.pdf_service import extract_text

router = APIRouter(prefix="/api/candidate", tags=["candidate"])

# Browsers sometimes send a generic type for drag-and-dropped files, so the
# filename extension is accepted as a fallback signal.
_ACCEPTED_CONTENT_TYPES = frozenset({"application/pdf", "application/octet-stream"})

_UNREADABLE_PDF = (
    "Could not extract enough text from the PDF. Make sure it is not a "
    "scanned or image-only document."
)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a resume against a job description",
    responses={
        400: {"description": "Empty file"},
        413: {"description": "File exceeds the size limit"},
        415: {"description": "Uploaded file is not a PDF"},
        422: {"description": "PDF could not be parsed, or its text is too short"},
    },
)
async def analyze_resume(
    resume: UploadFile = File(..., description="Candidate resume in PDF format"),
    job_description: str = Form(
        ...,
        min_length=MIN_JD_CHARS,
        max_length=MAX_JD_CHARS,
        description="Full job description text",
    ),
) -> AnalyzeResponse:
    started = time.perf_counter()

    filename = resume.filename or "resume.pdf"
    content_type = (resume.content_type or "").lower()
    if content_type not in _ACCEPTED_CONTENT_TYPES and not filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    if resume.size is not None and resume.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB size limit.",
        )

    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB size limit.",
        )

    loop = asyncio.get_running_loop()

    try:
        resume_text: str = await loop.run_in_executor(
            None, extract_text, file_bytes, filename
        )
    except Exception as exc:
        logger.error(f"PDF extraction failed for '{filename}': {exc}")
        raise HTTPException(status_code=422, detail=_UNREADABLE_PDF)

    if len(resume_text) < MIN_RESUME_CHARS:
        raise HTTPException(status_code=422, detail=_UNREADABLE_PDF)

    cache = get_cache()
    cache_key = cache.make_analysis_key(resume_text, job_description)

    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for '{filename}'")
        return _to_response(cached, (time.perf_counter() - started) * 1000)

    try:
        result: dict = await loop.run_in_executor(
            None, get_ml_service().analyze, resume_text, job_description
        )
    except Exception as exc:
        logger.exception(f"Analysis failed for '{filename}': {exc}")
        raise HTTPException(
            status_code=500, detail="Analysis failed. Please try again."
        )

    cache.set(cache_key, result)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        f"analyze | {filename} | score={result['score_pct']:.1f} | {elapsed_ms:.0f}ms"
    )
    return _to_response(result, elapsed_ms)


def _to_response(result: dict, elapsed_ms: float) -> AnalyzeResponse:
    """Map a raw analysis dict onto the public response model."""
    return AnalyzeResponse(
        score=result["score_pct"],
        label=result["label"],
        breakdown=ScoreBreakdown(**result["breakdown"]),
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        suggestions=result.get("suggestions", []),
        processing_time_ms=round(elapsed_ms, 1),
    )
