from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    skills_match: float = Field(..., ge=0.0, le=1.0, description="Skills overlap score")
    experience_relevance: float = Field(
        ..., ge=0.0, le=1.0, description="Experience similarity score"
    )
    education_fit: float = Field(
        ..., ge=0.0, le=1.0, description="Education alignment score"
    )
    keyword_alignment: float = Field(
        ..., ge=0.0, le=1.0, description="TF-IDF keyword overlap score"
    )


class ErrorResponse(BaseModel):
    detail: str
