"""
Recruiter-facing endpoints.

POST /api/recruiter/bulk-analyze
  - Accepts up to 50 PDFs + job description
  - Processes them concurrently (async + thread-pool)
  - Returns ranked candidate list; stores results in memory for later retrieval

GET  /api/recruiter/candidate/{id}
  - Returns the full analysis for one candidate by UUID

POST /api/recruiter/filter
  - Filters stored candidates by min_score, must_have_skills, experience_level
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

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

router = APIRouter(prefix="/api/recruiter", tags=["recruiter"])

_MAX_RESUMES = 50
_MAX_FILE_BYTES = 10 * 1024 * 1024

# Thread pool for CPU-bound ML work (8 workers saturates the GPU queue nicely)
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hirelens-ml")

# In-memory candidate store: {candidate_id: CandidateResult.model_dump()}
_candidate_store: dict[str, dict] = {}


# ── Bulk analyze ───────────────────────────────────────────────────────────────


@router.post(
    "/bulk-analyze",
    response_model=BulkAnalyzeResponse,
    summary="Rank multiple resumes against a single job description",
)
async def bulk_analyze(
    resumes: list[UploadFile] = File(..., description="Resume PDFs (max 50)"),
    job_description: str = Form(..., min_length=50, description="Job description text"),
) -> BulkAnalyzeResponse:
    t_start = time.perf_counter()

    if len(resumes) == 0:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")
    if len(resumes) > _MAX_RESUMES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {_MAX_RESUMES} resumes per request."
        )

    # Read all bytes eagerly (network I/O, do sequentially in async context)
    file_data: list[tuple[str, bytes]] = []
    for upload in resumes:
        data = await upload.read()
        if len(data) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413, detail=f"'{upload.filename}' exceeds 10 MB."
            )
        file_data.append((upload.filename or f"resume_{len(file_data)+1}.pdf", data))

    loop = asyncio.get_event_loop()
    ml = get_ml_service()
    cache = get_cache()

    async def _process_one(
        filename: str, pdf_bytes: bytes
    ) -> Optional[CandidateResult]:
        try:
            resume_text: str = await loop.run_in_executor(
                _executor, extract_text, pdf_bytes, filename
            )
            if len(resume_text) < 100:
                logger.warning(f"Skipping '{filename}': insufficient extracted text.")
                return None

            cache_key = cache.make_analysis_key(resume_text, job_description)
            result = cache.get(cache_key)
            if result is None:
                result = await loop.run_in_executor(
                    _executor, ml.analyze, resume_text, job_description
                )
                cache.set(cache_key, result)

            cid = str(uuid.uuid4())
            bd = result["breakdown"]
            candidate = CandidateResult(
                id=cid,
                filename=filename,
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
                rank=1,  # placeholder; reassigned after sorting
                experience_level=result.get("experience_level"),
            )
            _candidate_store[cid] = candidate.model_dump()
            return candidate

        except Exception as exc:
            logger.error(f"Failed to process '{filename}': {exc}")
            return None

    tasks = [_process_one(fn, data) for fn, data in file_data]
    raw_results = await asyncio.gather(*tasks)

    candidates = [c for c in raw_results if c is not None]
    candidates.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank
        _candidate_store[c.id]["rank"] = rank

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"bulk-analyze | {len(candidates)}/{len(resumes)} successful | {elapsed_ms:.0f}ms"
    )

    return BulkAnalyzeResponse(
        total=len(candidates),
        candidates=candidates,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ── Single candidate lookup ────────────────────────────────────────────────────


@router.get(
    "/candidate/{candidate_id}",
    response_model=CandidateResult,
    summary="Get full analysis for a specific candidate",
)
async def get_candidate(candidate_id: str) -> CandidateResult:
    stored = _candidate_store.get(candidate_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate '{candidate_id}' not found. "
            "Run /bulk-analyze first, then use the returned IDs.",
        )
    return CandidateResult(**stored)


# ── Filter ─────────────────────────────────────────────────────────────────────

_LEVEL_RANK = {"entry": 0, "mid": 1, "senior": 2, "lead": 3}


@router.post(
    "/filter",
    response_model=FilterResponse,
    summary="Filter and re-rank stored candidates",
)
async def filter_candidates(request: FilterRequest) -> FilterResponse:
    if not _candidate_store:
        raise HTTPException(
            status_code=404,
            detail="No candidates in store. Run /bulk-analyze first.",
        )

    min_level_rank = _LEVEL_RANK.get(request.experience_level or "", -1)
    must_have = [s.lower().strip() for s in request.must_have_skills]

    candidates: list[CandidateResult] = []
    for stored in _candidate_store.values():
        c = CandidateResult(**stored)

        # Score gate
        if c.score < request.min_score:
            continue

        # Must-have skills gate (match against lowercase matched_skills)
        if must_have:
            matched_lower = {s.lower() for s in c.matched_skills}
            if not all(skill in matched_lower for skill in must_have):
                continue

        # Experience level gate
        if request.experience_level:
            cand_level_rank = _LEVEL_RANK.get(c.experience_level or "entry", 0)
            if cand_level_rank < min_level_rank:
                continue

        candidates.append(c)

    candidates.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank

    return FilterResponse(total=len(candidates), candidates=candidates)
