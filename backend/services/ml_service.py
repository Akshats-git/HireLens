"""
Loads the ML stack once at startup and exposes the single scoring entry point
used by both routers.

Model imports are deferred to `startup()` so that importing this module — which
the test suite does with the service mocked — does not pull in torch and spaCy.
"""

import time
from typing import Optional

from loguru import logger

from src.config import get_section

# Lower bound in years for each tier, ordered highest tier first so the first
# threshold met wins.
_EXPERIENCE_TIERS: list[tuple[str, float]] = sorted(
    (
        (level, float(years))
        for level, years in get_section("scoring")["experience_levels"].items()
    ),
    key=lambda tier: tier[1],
    reverse=True,
)


def years_to_level(years: float) -> str:
    """Map years of experience to a seniority tier from configs/config.yaml."""
    for level, minimum in _EXPERIENCE_TIERS:
        if years >= minimum:
            return level
    return _EXPERIENCE_TIERS[-1][0]


class MLService:
    """Holds the scorer and skill extractor for the lifetime of the process."""

    def __init__(self) -> None:
        self._scorer = None
        self._extractor = None
        self._loaded = False

    def startup(self) -> None:
        """
        Load the models. Called once from the application lifespan.

        Raises:
            Exception: Propagated so the process fails fast rather than serving
                       requests that would all return 500.
        """
        logger.info("Loading ML models…")
        started = time.perf_counter()

        from src.features.ner import get_extractor
        from src.models.scorer import get_scorer

        self._scorer = get_scorer()
        self._extractor = get_extractor()

        # First inference compiles lazy code paths and allocates GPU buffers;
        # doing it here keeps that cost off the first real request.
        self._scorer.score(
            "Software engineer with 3 years of Python experience.",
            "Seeking a Python backend developer with REST API skills.",
        )

        self._loaded = True
        logger.info(f"ML models ready in {time.perf_counter() - started:.1f}s")

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """
        Score a resume against a job description.

        Args:
            resume_text: Extracted resume text.
            jd_text: Job description text.

        Returns:
            MatchResult.to_dict() plus an 'experience_level' key, which is None
            when the seniority tier could not be determined.

        Raises:
            RuntimeError: If called before `startup()`.
        """
        if not self._loaded:
            raise RuntimeError("ML models are not loaded — call startup() first.")

        output = self._scorer.score(resume_text, jd_text).to_dict()

        # Seniority is supplementary; a failure here should not sink the score.
        try:
            extraction = self._extractor.extract(resume_text)
            output["experience_level"] = years_to_level(extraction.experience_years)
        except Exception as exc:
            logger.warning(f"Could not determine experience level: {exc}")
            output["experience_level"] = None

        return output

    @property
    def is_ready(self) -> bool:
        """True once the models have finished loading."""
        return self._loaded


# ── Singleton ─────────────────────────────────────────────────────────────────

_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Return the shared MLService instance."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
