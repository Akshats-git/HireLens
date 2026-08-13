"""
Recruiter endpoints.

POST /api/recruiter/bulk-analyze
    Score up to `MAX_BULK_RESUMES` PDFs against one job description and return
    them ranked, along with a batch ID.

POST /api/recruiter/filter
    Re-filter and re-rank a previously analysed batch.

GET  /api/recruiter/batches/{batch_id}/candidates/{candidate_id}
    Retrieve one candidate's full analysis.
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ..config import (
    BULK_WORKER_THREADS,
    MAX_BULK_RESUMES,
    MAX_JD_CHARS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MIN_JD_CHARS,
    MIN_RESUME_CHARS,
)
from ..schemas.common import ScoreBreakdown
from ..schemas.recruiter import (
    BulkAnalyzeResponse,
    CandidateResult,
    FilterRequest,
    FilterResponse,
)
from ..services.cache_service import get_cache
from ..services.ml_service import get_ml_service
from ..services.pdf_service import extract_text
from ..services.result_store import get_result_store

router = APIRouter(prefix="/api/recruiter", tags=["recruiter"])

# Scoring is CPU/GPU-bound and releases the GIL inside torch, so it runs on a
# dedicated pool rather than the default executor shared with other blocking work.
_executor = ThreadPoolExecutor(
    max_workers=BULK_WORKER_THREADS, thread_name_prefix="hirelens-ml"
)

_BATCH_NOT_FOUND = (
    "Batch not found or expired. Re-run /bulk-analyze to obtain a new batch ID."
)


# ── Bulk analyze ───────────────────────────────────────────────────────────────


@router.post(
    "/bulk-analyze",
    response_model=BulkAnalyzeResponse,
    summary="Rank multiple resumes against a single job description",
    responses={
        400: {"description": "No files uploaded, or more than the per-request limit"},
        413: {"description": "A file exceeds the size limit"},
    },
)
async def bulk_analyze(
    resumes: list[UploadFile] = File(
        ..., description=f"Resume PDFs (max {MAX_BULK_RESUMES})"
    ),
    job_description: str = Form(
        ...,
        min_length=MIN_JD_CHARS,
        max_length=MAX_JD_CHARS,
        description="Job description text",
    ),
) -> BulkAnalyzeResponse:
    started = time.perf_counter()

    if not resumes:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")
    if len(resumes) > MAX_BULK_RESUMES:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_BULK_RESUMES} resumes may be uploaded per request.",
        )

    uploads = await _read_uploads(resumes)

    loop = asyncio.get_running_loop()
    ml = get_ml_service()
    cache = get_cache()

    async def score_one(filename: str, pdf_bytes: bytes) -> Optional[CandidateResult]:
        try:
            resume_text: str = await loop.run_in_executor(
                _executor, extract_text, pdf_bytes, filename
            )
            if len(resume_text) < MIN_RESUME_CHARS:
                logger.warning(f"Skipping '{filename}': insufficient extracted text.")
                return None

            cache_key = cache.make_analysis_key(resume_text, job_description)
            result = cache.get(cache_key)
            if result is None:
                result = await loop.run_in_executor(
                    _executor, ml.analyze, resume_text, job_description
                )
                cache.set(cache_key, result)

            return _to_candidate(result, filename)
        except Exception as exc:
            logger.error(f"Failed to process '{filename}': {exc}")
            return None

    scored = await asyncio.gather(*(score_one(name, data) for name, data in uploads))

    candidates = _rank([c for c in scored if c is not None])
    batch_id = get_result_store().create([c.model_dump() for c in candidates])

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        f"bulk-analyze | batch={batch_id} | {len(candidates)}/{len(resumes)} scored "
        f"| {elapsed_ms:.0f}ms"
    )

    return BulkAnalyzeResponse(
        batch_id=batch_id,
        total=len(candidates),
        submitted=len(resumes),
        candidates=candidates,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ── Filter ─────────────────────────────────────────────────────────────────────


@router.post(
    "/filter",
    response_model=FilterResponse,
    summary="Filter and re-rank the candidates of a previous batch",
    responses={404: {"description": "Unknown or expired batch ID"}},
)
async def filter_candidates(request: FilterRequest) -> FilterResponse:
    stored = get_result_store().get(request.batch_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=_BATCH_NOT_FOUND)

    required_skills = {skill.lower().strip() for skill in request.must_have_skills}
    required_skills.discard("")
    minimum_rank = request.experience_level.rank if request.experience_level else None

    matches: list[CandidateResult] = []
    for record in stored:
        candidate = CandidateResult(**record)

        if candidate.score < request.min_score:
            continue

        if required_skills:
            candidate_skills = {skill.lower() for skill in candidate.matched_skills}
            if not required_skills.issubset(candidate_skills):
                continue

        if minimum_rank is not None:
            if candidate.experience_level is None:
                continue
            if candidate.experience_level.rank < minimum_rank:
                continue

        matches.append(candidate)

    return FilterResponse(total=len(matches), candidates=_rank(matches))


# ── Single candidate ───────────────────────────────────────────────────────────


@router.get(
    "/batches/{batch_id}/candidates/{candidate_id}",
    response_model=CandidateResult,
    summary="Get the full analysis for one candidate",
    responses={404: {"description": "Unknown batch or candidate ID"}},
)
async def get_candidate(batch_id: str, candidate_id: str) -> CandidateResult:
    record = get_result_store().get_candidate(batch_id, candidate_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate '{candidate_id}' not found in this batch. {_BATCH_NOT_FOUND}",
        )
    return CandidateResult(**record)


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _read_uploads(resumes: list[UploadFile]) -> list[tuple[str, bytes]]:
    """
    Read every upload into memory, rejecting oversized files.

    Size is checked from the upload's declared length before reading so an
    oversized file is refused rather than buffered in full first.
    """
    uploads: list[tuple[str, bytes]] = []
    for index, upload in enumerate(resumes, start=1):
        filename = upload.filename or f"resume_{index}.pdf"
        if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"'{filename}' exceeds the {MAX_UPLOAD_SIZE_MB} MB size limit.",
            )

        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"'{filename}' exceeds the {MAX_UPLOAD_SIZE_MB} MB size limit.",
            )
        uploads.append((filename, data))
    return uploads


def _to_candidate(result: dict, filename: str) -> CandidateResult:
    """Build a CandidateResult from a raw analysis dict. Rank is assigned later."""
    breakdown = result["breakdown"]
    return CandidateResult(
        id=str(uuid.uuid4()),
        filename=filename,
        score=result["score_pct"],
        label=result["label"],
        breakdown=ScoreBreakdown(**breakdown),
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        suggestions=result.get("suggestions", []),
        rank=1,
        experience_level=result.get("experience_level"),
    )


def _rank(candidates: list[CandidateResult]) -> list[CandidateResult]:
    """Sort candidates by score descending and assign contiguous ranks."""
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    for position, candidate in enumerate(ordered, start=1):
        candidate.rank = position
    return ordered
