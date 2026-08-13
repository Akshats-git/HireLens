from enum import Enum

from pydantic import BaseModel, Field


class ExperienceLevel(str, Enum):
    """Seniority tiers, ordered entry through lead."""

    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"

    @property
    def rank(self) -> int:
        """Ordinal position, for comparing one tier against another."""
        return list(ExperienceLevel).index(self)


class ScoreBreakdown(BaseModel):
    """The four weighted components behind an overall match score."""

    skills_match: float = Field(
        ..., ge=0.0, le=1.0, description="Skills overlap, exact and semantic"
    )
    experience_relevance: float = Field(
        ..., ge=0.0, le=1.0, description="Experience similarity and years alignment"
    )
    education_fit: float = Field(
        ..., ge=0.0, le=1.0, description="Degree level against the role's requirement"
    )
    keyword_alignment: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Resume coverage of the posting's most emphasised terms",
    )


class ErrorResponse(BaseModel):
    detail: str
