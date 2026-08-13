from typing import Optional

from pydantic import BaseModel, Field

from .common import ExperienceLevel, ScoreBreakdown


class CandidateResult(BaseModel):
    """One resume's scored result within a bulk analysis."""

    id: str = Field(..., description="Unique candidate ID (UUID)")
    filename: str
    score: float = Field(..., ge=0.0, le=100.0)
    label: str
    breakdown: ScoreBreakdown
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    rank: int = Field(
        ..., ge=1, description="Rank among the returned candidates (1 = best)"
    )
    experience_level: Optional[ExperienceLevel] = Field(
        None, description="Seniority tier inferred from the resume"
    )


class BulkAnalyzeResponse(BaseModel):
    batch_id: str = Field(
        ...,
        description="Identifier for this upload, required to filter or look up "
        "its candidates later",
    )
    total: int = Field(..., description="Number of resumes scored successfully")
    submitted: int = Field(..., description="Number of resumes uploaded")
    candidates: list[CandidateResult]
    processing_time_ms: float


class FilterRequest(BaseModel):
    """Server-side filter over a previously analysed batch."""

    batch_id: str = Field(..., description="Batch ID returned by /bulk-analyze")
    min_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Minimum match score (0–100)"
    )
    must_have_skills: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Every candidate must match all of these skills",
    )
    experience_level: Optional[ExperienceLevel] = Field(
        None, description="Minimum seniority tier"
    )


class FilterResponse(BaseModel):
    total: int
    candidates: list[CandidateResult]
