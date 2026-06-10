"""
ML service wrapper.

Loads all heavy ML components once at startup and exposes a single
`analyze(resume_text, jd_text)` method used by both routers.
"""

import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Experience level thresholds (years)
_EXP_LEVELS = [
    ("lead", 10.0),
    ("senior", 5.0),
    ("mid", 2.0),
    ("entry", 0.0),
]


def _years_to_level(years: float) -> str:
    for label, threshold in _EXP_LEVELS:
        if years >= threshold:
            return label
    return "entry"


class MLService:
    """Singleton wrapper around scorer, embeddings, and NER."""

    def __init__(self) -> None:
        self._scorer = None
        self._extractor = None
        self._loaded = False

    def startup(self) -> None:
        """Load all ML models. Must be called once before serving requests."""
        logger.info("Loading ML models…")
        t0 = time.perf_counter()

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.models.scorer import get_scorer
        from src.features.ner import get_extractor

        self._scorer = get_scorer()
        self._extractor = get_extractor()

        # Warm-up pass to JIT-compile any lazy code paths
        self._scorer.score(
            "Software engineer with 3 years Python experience.",
            "Seeking a Python backend developer with REST API skills.",
        )

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info(f"ML models ready in {elapsed:.1f}s")

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """
        Full match scoring pipeline.

        Returns:
            dict matching MatchResult.to_dict() plus an 'experience_level' key.
        """
        if not self._loaded:
            raise RuntimeError("ML models not loaded — call startup() first.")

        result = self._scorer.score(resume_text, jd_text)
        output = result.to_dict()

        # Attach experience level derived from NER
        try:
            ner = self._extractor.extract(resume_text)
            output["experience_level"] = _years_to_level(ner.experience_years)
        except Exception:
            output["experience_level"] = None

        return output

    @property
    def is_ready(self) -> bool:
        return self._loaded


# ── Singleton ─────────────────────────────────────────────────────────────────

_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
