from typing import Optional

from pydantic import BaseModel, Field

from .common import ScoreBreakdown


class CandidateResult(BaseModel):
    id: str = Field(..., description="Unique candidate ID (UUID)")
    filename: str
    score: float = Field(..., ge=0.0, le=100.0)
    label: str
    breakdown: ScoreBreakdown
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    rank: int = Field(
        ..., ge=1, description="Rank among all processed candidates (1 = best)"
    )
    experience_level: Optional[str] = Field(
        None, description="entry / mid / senior / lead"
    )


class BulkAnalyzeResponse(BaseModel):
    total: int
    candidates: list[CandidateResult]
    processing_time_ms: float


class FilterRequest(BaseModel):
    min_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Minimum match score (0–100)"
    )
    must_have_skills: list[str] = Field(
        default_factory=list, description="Skills every candidate must have"
    )
    experience_level: Optional[str] = Field(
        None,
        description="Minimum experience tier: entry | mid | senior | lead",
    )


class FilterResponse(BaseModel):
    total: int
    candidates: list[CandidateResult]
