from pydantic import BaseModel, Field

from .common import ScoreBreakdown


class AnalyzeResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0, description="Overall match score (0–100)")
    label: str = Field(..., description="Excellent / Good / Fair / Poor")
    breakdown: ScoreBreakdown
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
