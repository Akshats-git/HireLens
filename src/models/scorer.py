"""
Four-component match scorer.

Combines, using the weights in configs/config.yaml:
  1. Skills match       — Jaccard overlap blended with semantic similarity
  2. Experience relevance — semantic similarity plus a years-of-experience delta
  3. Education fit      — degree-hierarchy gap between resume and posting
  4. Keyword alignment  — resume coverage of the posting's most emphasised terms

Produces a MatchResult carrying the overall score, the per-component breakdown,
a human-readable label, and improvement suggestions.
"""

import re
from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np
from loguru import logger

from src.config import get_section
from src.features.patterns import (
    DEGREE_RANK,
    detect_education_level,
    extract_experience_years,
    required_experience_years,
)

# Weighting between exact skill overlap and embedding similarity.
JACCARD_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.40

# Neutral score used when a component has nothing to compare.
NEUTRAL_SCORE = 0.5

# Education fit by how many degree levels the resume falls short.
_EDUCATION_GAP_SCORES = {0: 1.0, -1: 0.7, -2: 0.4}
_EDUCATION_GAP_FLOOR = 0.2

# A posting that omits its education requirement is treated as wanting a degree.
_ASSUMED_JD_EDUCATION = "bachelors"

# Experience deltas, in years, translated into a bonus on the semantic score.
_EXPERIENCE_OVERQUALIFIED_YEARS = 3
_EXPERIENCE_MATCH_BONUS = 0.1
_EXPERIENCE_SHORTFALL_PER_YEAR = 0.05
_EXPERIENCE_MAX_PENALTY = -0.2

# Suggestion trigger points.
_LOW_SKILLS = 0.5
_VERY_LOW_SKILLS = 0.3
_LOW_EXPERIENCE = 0.5
_LOW_EDUCATION = 0.5
_LOW_KEYWORDS = 0.4
_STRONG_OVERALL = 0.85

MAX_MISSING_SKILLS_REPORTED = 10
MAX_MISSING_SKILLS_SUGGESTED = 5

_STOPWORDS = frozenset(
    """
    a an the and or but in on at to for of with by from is are was were be been
    being have has had do does did will would could should may might shall can
    need dare ought used we you i he she it they them their our your this that
    these those what which who whom not no nor so yet both either neither
    """.split()
)

# Words of two or more characters, plus dotted and symbol-suffixed technology
# names. The trailing symbols need their own branch: a closing \b cannot follow
# '+' or '#', so a single pattern ending in \b drops "c++" and "c#" outright.
_TOKEN_PATTERN = re.compile(r"\b[a-z](?:[a-z0-9]+(?:\.[a-z0-9]+)*|[+#]+)")


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class MatchResult:
    """A complete resume-to-posting match, with component scores and advice."""

    final_score: float  # 0.0–1.0
    score_pct: float  # 0–100, for display
    label: str  # Excellent / Good / Fair / Poor
    skills_match: float
    experience_relevance: float
    education_fit: float
    keyword_alignment: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 4),
            "score_pct": round(self.score_pct, 1),
            "label": self.label,
            "breakdown": {
                "skills_match": round(self.skills_match, 4),
                "experience_relevance": round(self.experience_relevance, 4),
                "education_fit": round(self.education_fit, 4),
                "keyword_alignment": round(self.keyword_alignment, 4),
            },
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills[:MAX_MISSING_SKILLS_REPORTED],
            "suggestions": self.suggestions,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Lowercase the text and return its content words, minus stopwords."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Dot product of two L2-normalised embeddings, or None if either is absent."""
    if a is None or b is None:
        return None
    return float(np.clip(np.dot(np.atleast_1d(a), np.atleast_1d(b)), 0.0, 1.0))


# ── Component scorers ─────────────────────────────────────────────────────────


def score_skills_match(
    resume_skills: set[str],
    jd_skills: set[str],
    resume_emb: np.ndarray | None = None,
    jd_emb: np.ndarray | None = None,
) -> tuple[float, list[str], list[str]]:
    """
    Score skill overlap, blending exact matches with semantic similarity.

    Jaccard captures literal overlap; the embedding term rewards paraphrases such
    as 'ML' against 'machine learning'. The blend applies only when both
    embeddings are present, so a half-supplied pair does not score as if the
    semantic component were zero.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.
        resume_emb: Resume embedding, for the semantic term.
        jd_emb: Job description embedding, for the semantic term.

    Returns:
        (score in [0, 1], matched skills, skills the posting wants but the resume lacks)
    """
    if not resume_skills and not jd_skills:
        return NEUTRAL_SCORE, [], []

    union = resume_skills | jd_skills
    jaccard = len(resume_skills & jd_skills) / len(union) if union else 0.0

    semantic = _cosine(resume_emb, jd_emb)
    score = (
        jaccard
        if semantic is None
        else JACCARD_WEIGHT * jaccard + SEMANTIC_WEIGHT * semantic
    )

    return (
        float(np.clip(score, 0.0, 1.0)),
        sorted(resume_skills & jd_skills),
        sorted(jd_skills - resume_skills),
    )


def score_experience_relevance(
    resume_text: str,
    jd_text: str,
    resume_emb: np.ndarray | None = None,
    jd_emb: np.ndarray | None = None,
) -> float:
    """
    Score experience fit from overall semantic similarity plus a years delta.

    Experience sections dominate resume length, so whole-document similarity is a
    reasonable proxy. The years adjustment rewards meeting the stated requirement
    and penalises falling short, capped so it cannot dominate the score.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.
        resume_emb: Resume embedding.
        jd_emb: Job description embedding.

    Returns:
        Score in [0, 1].
    """
    semantic = _cosine(resume_emb, jd_emb)
    if semantic is None:
        semantic = NEUTRAL_SCORE

    resume_years = extract_experience_years(resume_text)
    required_years = required_experience_years(jd_text)

    bonus = 0.0
    if resume_years and required_years is not None:
        gap = resume_years - required_years
        if gap >= 0:
            bonus = (
                _EXPERIENCE_MATCH_BONUS
                if gap <= _EXPERIENCE_OVERQUALIFIED_YEARS
                else 0.0
            )
        else:
            bonus = max(_EXPERIENCE_MAX_PENALTY, gap * _EXPERIENCE_SHORTFALL_PER_YEAR)

    return float(np.clip(semantic + bonus, 0.0, 1.0))


def score_education_fit(resume_text: str, jd_text: str) -> float:
    """
    Score the degree-hierarchy gap between the resume and the posting.

    Returns 1.0 when the candidate meets or exceeds the requirement, tapering as
    the shortfall widens.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.

    Returns:
        Score in [0, 1].
    """
    resume_rank = DEGREE_RANK[detect_education_level(resume_text)]
    jd_rank = DEGREE_RANK[
        detect_education_level(jd_text, default=_ASSUMED_JD_EDUCATION)
    ]

    gap = resume_rank - jd_rank
    if gap > 0:
        gap = 0
    return _EDUCATION_GAP_SCORES.get(gap, _EDUCATION_GAP_FLOOR)


def keyword_weights(tokens: list[str]) -> dict[str, float]:
    """
    Weight a document's terms by how much the document emphasises them.

    Weighting is sublinear term frequency, log(1 + count). An IDF factor is
    deliberately absent: document frequency across a resume/posting pair can only
    be 1 or 2, so IDF collapses to log(2) for terms in one document and 0 for
    terms in both — which zeroes out precisely the overlapping terms this score
    exists to reward. Meaningful IDF would need a background corpus.

    Args:
        tokens: Tokens of the document being weighted.

    Returns:
        Mapping of token to weight.
    """
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return {token: log(1 + count) for token, count in counts.items()}


def score_keyword_alignment(resume_text: str, jd_text: str, top_k: int) -> float:
    """
    Score how much of the posting's keyword weight the resume covers.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.
        top_k: Number of highest-weighted posting keywords to consider.

    Returns:
        Score in [0, 1]: 1.0 when the resume mentions every top keyword.
    """
    resume_tokens = tokenize(resume_text)
    jd_tokens = tokenize(jd_text)
    if not jd_tokens:
        return NEUTRAL_SCORE

    weights = keyword_weights(jd_tokens)
    top_keywords = sorted(weights.items(), key=lambda item: item[1], reverse=True)[
        :top_k
    ]

    total_weight = sum(weight for _, weight in top_keywords)
    if total_weight <= 0:
        return NEUTRAL_SCORE

    resume_vocabulary = set(resume_tokens)
    matched_weight = sum(
        weight for keyword, weight in top_keywords if keyword in resume_vocabulary
    )
    return float(np.clip(matched_weight / total_weight, 0.0, 1.0))


# ── Suggestions ───────────────────────────────────────────────────────────────


def generate_suggestions(result: MatchResult, jd_text: str) -> list[str]:
    """
    Derive actionable advice from the weakest components of a match.

    Args:
        result: A scored MatchResult.
        jd_text: Job description text, used to name the required degree.

    Returns:
        Suggestions ordered by expected impact; never empty.
    """
    suggestions: list[str] = []

    if result.skills_match < _LOW_SKILLS and result.missing_skills:
        top_missing = ", ".join(result.missing_skills[:MAX_MISSING_SKILLS_SUGGESTED])
        suggestions.append(f"Add missing skills to your resume: {top_missing}.")

    if result.skills_match < _VERY_LOW_SKILLS:
        suggestions.append(
            "Your skill set has low overlap with this role. Consider upskilling "
            "in the required technologies before applying."
        )

    if result.experience_relevance < _LOW_EXPERIENCE:
        suggestions.append(
            "Emphasise relevant experience more prominently. Rewrite bullet "
            "points to mirror the job description's language."
        )

    if result.education_fit < _LOW_EDUCATION:
        required = detect_education_level(jd_text)
        if required != "none":
            suggestions.append(
                f"This role prefers candidates with a {required} qualification. "
                "Highlight certifications or equivalent experience to compensate."
            )

    if result.keyword_alignment < _LOW_KEYWORDS:
        suggestions.append(
            "Your resume is missing key terms from the job posting. Mirror the "
            "posting's exact phrasing for ATS compatibility."
        )

    if not suggestions:
        suggestions.append(
            "Strong match — apply with confidence."
            if result.final_score >= _STRONG_OVERALL
            else "Good match overall. Tailor your summary section to this role."
        )

    return suggestions


# ── Scorer ────────────────────────────────────────────────────────────────────


class MatchScorer:
    """
    Scores a resume against a job description across four weighted components.

    Weights, label thresholds, and the keyword cutoff come from
    configs/config.yaml and are read once at construction.
    """

    def __init__(self) -> None:
        scoring = get_section("scoring")
        self._weights = scoring["weights"]
        self._thresholds = scoring["thresholds"]
        self._top_k_keywords = get_section("similarity")["top_k_keywords"]

        total = sum(self._weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Scoring weights sum to {total:.4f}, not 1.0 — overall scores "
                "will not span the full 0–100 range."
            )

    def _label(self, score_pct: float) -> str:
        if score_pct >= self._thresholds["excellent"]:
            return "Excellent"
        if score_pct >= self._thresholds["good"]:
            return "Good"
        if score_pct >= self._thresholds["fair"]:
            return "Fair"
        return "Poor"

    def _resolve_embeddings(
        self,
        resume_text: str,
        jd_text: str,
        resume_emb: np.ndarray | None,
        jd_emb: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if resume_emb is not None and jd_emb is not None:
            return resume_emb, jd_emb
        try:
            from src.features.embeddings import get_model

            model = get_model()
            if resume_emb is None:
                resume_emb = model.encode(resume_text)
            if jd_emb is None:
                jd_emb = model.encode(jd_text)
        except Exception as exc:
            logger.warning(f"Embeddings unavailable, scoring on text alone: {exc}")
            return None, None
        return resume_emb, jd_emb

    def _resolve_skills(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: set[str] | None,
        jd_skills: set[str] | None,
    ) -> tuple[set[str], set[str]]:
        if resume_skills is not None and jd_skills is not None:
            return resume_skills, jd_skills
        try:
            from src.features.ner import get_extractor

            extractor = get_extractor()
            if resume_skills is None:
                resume_skills = set(extractor.extract(resume_text).technical_skills)
            if jd_skills is None:
                jd_skills = set(extractor.extract(jd_text).technical_skills)
        except Exception as exc:
            logger.warning(f"Skill extraction unavailable, using empty sets: {exc}")
            return set(), set()
        return resume_skills, jd_skills

    def score(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: set[str] | None = None,
        jd_skills: set[str] | None = None,
        resume_emb: np.ndarray | None = None,
        jd_emb: np.ndarray | None = None,
    ) -> MatchResult:
        """
        Score a resume against a job description.

        Any pre-computed input may be supplied to skip the corresponding model
        call; anything omitted is computed on demand.

        Args:
            resume_text: Raw resume text.
            jd_text: Raw job description text.
            resume_skills: Pre-extracted resume skills.
            jd_skills: Pre-extracted job description skills.
            resume_emb: Pre-computed resume embedding.
            jd_emb: Pre-computed job description embedding.

        Returns:
            A fully populated MatchResult.
        """
        resume_emb, jd_emb = self._resolve_embeddings(
            resume_text, jd_text, resume_emb, jd_emb
        )
        resume_skills, jd_skills = self._resolve_skills(
            resume_text, jd_text, resume_skills, jd_skills
        )

        skills, matched, missing = score_skills_match(
            resume_skills, jd_skills, resume_emb, jd_emb
        )
        experience = score_experience_relevance(
            resume_text, jd_text, resume_emb, jd_emb
        )
        education = score_education_fit(resume_text, jd_text)
        keywords = score_keyword_alignment(resume_text, jd_text, self._top_k_keywords)

        final = float(
            np.clip(
                self._weights["skills_match"] * skills
                + self._weights["experience_relevance"] * experience
                + self._weights["education_fit"] * education
                + self._weights["keyword_alignment"] * keywords,
                0.0,
                1.0,
            )
        )
        score_pct = round(final * 100, 1)

        result = MatchResult(
            final_score=final,
            score_pct=score_pct,
            label=self._label(score_pct),
            skills_match=skills,
            experience_relevance=experience,
            education_fit=education,
            keyword_alignment=keywords,
            matched_skills=matched,
            missing_skills=missing,
        )
        result.suggestions = generate_suggestions(result, jd_text)
        return result


# ── Module-level singleton ────────────────────────────────────────────────────

_default_scorer: MatchScorer | None = None


def get_scorer() -> MatchScorer:
    """Return the shared MatchScorer instance."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = MatchScorer()
    return _default_scorer
