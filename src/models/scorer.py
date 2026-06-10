"""
4-component match scorer for HireLens.

Computes:
  1. Skills Match Score      — Jaccard + semantic similarity of skill sets   (weight: 0.40)
  2. Experience Relevance    — Semantic similarity of experience sections     (weight: 0.30)
  3. Education Fit Score     — Rule-based degree hierarchy + semantic         (weight: 0.15)
  4. Keyword Alignment Score — TF-IDF weighted keyword overlap                (weight: 0.15)

Returns a MatchResult with the final score, per-component breakdown,
score label (Excellent/Good/Fair/Poor), and improvement suggestions.
"""

import re
import sys
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Degree hierarchy ──────────────────────────────────────────────────────────

_DEGREE_ORDER = [
    "none",
    "bootcamp",
    "certification",
    "associate",
    "diploma",
    "bachelors",
    "masters",
    "phd",
]
_DEGREE_RANK = {d: i for i, d in enumerate(_DEGREE_ORDER)}

_EDU_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("phd", re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I)),
    (
        "masters",
        re.compile(r"\b(m\.?s\.?|m\.?eng\.?|m\.?b\.?a\.?|master(?:\'?s)?)\b", re.I),
    ),
    (
        "bachelors",
        re.compile(
            r"\b(b\.?s\.?|b\.?e\.?|b\.?tech\.?|bachelor(?:\'?s)?|undergraduate)\b", re.I
        ),
    ),
    ("associate", re.compile(r"\b(associate(?:\'?s)?)\b", re.I)),
    ("diploma", re.compile(r"\b(diploma)\b", re.I)),
    ("bootcamp", re.compile(r"\b(bootcamp|nanodegree|coding school)\b", re.I)),
    ("certification", re.compile(r"\b(certificate|certified|certification)\b", re.I)),
]

_EXP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s+)?(?:experience|exp\.?)",
    re.I,
)

# Simple stopwords for TF-IDF
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "we",
    "you",
    "i",
    "he",
    "she",
    "it",
    "they",
    "them",
    "their",
    "our",
    "your",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "not",
    "no",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
}


# ── Output dataclass ──────────────────────────────────────────────────────────


@dataclass
class MatchResult:
    """Complete match result with scores, label, and suggestions."""

    final_score: float  # 0.0–1.0
    score_pct: float  # 0–100 (display)
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
            "missing_skills": self.missing_skills[:10],
            "suggestions": self.suggestions,
        }


# ── Individual scorers ────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization, removing stopwords and short tokens."""
    words = re.findall(r"\b[a-z][a-z0-9+#.]{1,}\b", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def score_skills_match(
    resume_skills: set[str],
    jd_skills: set[str],
    resume_emb: np.ndarray | None = None,
    jd_emb: np.ndarray | None = None,
) -> tuple[float, list[str], list[str]]:
    """
    Skills Match Score: weighted average of Jaccard similarity and semantic similarity.

    Jaccard captures exact skill overlap; semantic similarity catches
    paraphrases (e.g., 'ML' ↔ 'machine learning').

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.
        resume_emb: Embedding of resume text (for semantic component).
        jd_emb: Embedding of JD text (for semantic component).

    Returns:
        (score, matched_skills, missing_skills)
    """
    if not resume_skills and not jd_skills:
        base = 0.5
        return base, [], []

    intersection = resume_skills & jd_skills
    union = resume_skills | jd_skills
    jaccard = len(intersection) / len(union) if union else 0.0

    # Semantic component
    semantic = 0.0
    if resume_emb is not None and jd_emb is not None:
        a = np.atleast_1d(resume_emb)
        b = np.atleast_1d(jd_emb)
        # Both should already be L2-normalised
        semantic = float(np.clip(np.dot(a, b), 0.0, 1.0))

    # Blend: 60% Jaccard (exact), 40% semantic (fuzzy)
    if resume_emb is not None:
        score = 0.60 * jaccard + 0.40 * semantic
    else:
        score = jaccard

    matched = sorted(intersection)
    missing = sorted(jd_skills - resume_skills)
    return float(np.clip(score, 0.0, 1.0)), matched, missing


def score_experience_relevance(
    resume_text: str,
    jd_text: str,
    resume_emb: np.ndarray | None = None,
    jd_emb: np.ndarray | None = None,
) -> float:
    """
    Experience Relevance Score: semantic similarity of experience sections
    plus a years-of-experience alignment bonus.

    Args:
        resume_text: Full resume text (section detection done internally).
        jd_text: Full job description text.
        resume_emb: Full resume embedding.
        jd_emb: Full JD embedding.

    Returns:
        Score in [0, 1].
    """
    # Semantic similarity of full texts (experience dominates resume length)
    semantic = 0.5
    if resume_emb is not None and jd_emb is not None:
        semantic = float(
            np.clip(np.dot(np.atleast_1d(resume_emb), np.atleast_1d(jd_emb)), 0.0, 1.0)
        )

    # Years-of-experience bonus/penalty
    resume_years_matches = _EXP_PATTERN.findall(resume_text)
    jd_years_matches = _EXP_PATTERN.findall(jd_text)

    exp_bonus = 0.0
    if resume_years_matches and jd_years_matches:
        resume_yrs = max(float(y) for y in resume_years_matches)
        jd_yrs = float(jd_years_matches[0])
        gap = resume_yrs - jd_yrs
        if gap >= 0:
            exp_bonus = 0.1 if gap <= 3 else 0.0
        else:
            exp_bonus = max(-0.2, gap * 0.05)

    return float(np.clip(semantic + exp_bonus, 0.0, 1.0))


def _detect_education(text: str) -> str:
    """Detect the highest education level mentioned in text."""
    for level, pattern in _EDU_PATTERNS:
        if pattern.search(text):
            return level
    return "none"


def score_education_fit(resume_text: str, jd_text: str) -> float:
    """
    Education Fit Score: rule-based degree hierarchy comparison.

    Returns 1.0 if resume meets or exceeds JD requirement,
    decreasing penalty as the gap widens.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.

    Returns:
        Score in [0, 1].
    """
    resume_edu = _detect_education(resume_text)
    jd_edu = _detect_education(jd_text)

    resume_rank = _DEGREE_RANK.get(resume_edu, 0)
    jd_rank = _DEGREE_RANK.get(jd_edu, 4)  # default to bachelors if JD doesn't mention

    gap = resume_rank - jd_rank
    if gap >= 0:
        return 1.0
    elif gap == -1:
        return 0.7
    elif gap == -2:
        return 0.4
    else:
        return 0.2


def _compute_tfidf_weights(
    tokens: list[str], corpus_tokens: list[str]
) -> dict[str, float]:
    """
    Compute simplified TF-IDF weights for tokens in a two-document corpus.

    Args:
        tokens: Tokens from the target document.
        corpus_tokens: Tokens from the other document (for IDF computation).

    Returns:
        Dict mapping token → TF-IDF weight.
    """
    tf: dict[str, float] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    total = max(len(tokens), 1)
    tf = {k: v / total for k, v in tf.items()}

    # IDF: log(2 / (1 + df)) where df ∈ {0, 1, 2}
    corpus_set = set(corpus_tokens)
    weights: dict[str, float] = {}
    for tok, freq in tf.items():
        df = (1 if tok in corpus_set else 0) + 1  # +1 for the current doc
        idf = log(2.0 / df)
        weights[tok] = freq * idf

    return weights


def score_keyword_alignment(resume_text: str, jd_text: str) -> float:
    """
    Keyword Alignment Score: TF-IDF weighted overlap of top-K keywords.

    Identifies the most important keywords in the JD and checks what
    fraction (by TF-IDF weight) appear in the resume.

    Args:
        resume_text: Full resume text.
        jd_text: Full job description text.

    Returns:
        Score in [0, 1].
    """
    cfg = _load_config()
    top_k = cfg["similarity"]["top_k_keywords"]

    resume_tokens = _tokenize(resume_text)
    jd_tokens = _tokenize(jd_text)

    if not jd_tokens:
        return 0.5

    jd_weights = _compute_tfidf_weights(jd_tokens, resume_tokens)

    # Select top-K keywords from JD
    sorted_kw = sorted(jd_weights.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if not sorted_kw:
        return 0.5

    resume_set = set(resume_tokens)
    total_weight = sum(w for _, w in sorted_kw)
    if total_weight <= 0:
        return 0.5

    matched_weight = sum(w for kw, w in sorted_kw if kw in resume_set)
    return float(np.clip(matched_weight / total_weight, 0.0, 1.0))


# ── Suggestion generator ──────────────────────────────────────────────────────


def generate_suggestions(result: "MatchResult", jd_text: str) -> list[str]:
    """
    Generate actionable improvement suggestions based on score gaps.

    Args:
        result: MatchResult with component scores and skill lists.
        jd_text: Job description text for context.

    Returns:
        List of suggestion strings, ordered by impact.
    """
    suggestions = []

    if result.skills_match < 0.5 and result.missing_skills:
        top_missing = result.missing_skills[:5]
        suggestions.append(
            f"Add missing skills to your resume: {', '.join(top_missing)}."
        )

    if result.skills_match < 0.3:
        suggestions.append(
            "Your skill set has low overlap with this role. "
            "Consider upskilling in the required technologies before applying."
        )

    if result.experience_relevance < 0.5:
        suggestions.append(
            "Emphasise relevant experience more prominently. "
            "Rewrite bullet points to mirror the job description's language."
        )

    if result.education_fit < 0.5:
        jd_edu = _detect_education(jd_text)
        if jd_edu != "none":
            suggestions.append(
                f"This role prefers {jd_edu} degree holders. "
                "Highlight certifications or equivalent experience to compensate."
            )

    if result.keyword_alignment < 0.4:
        suggestions.append(
            "Your resume is missing key terms from the job posting. "
            "Mirror the JD's exact phrasing for ATS compatibility."
        )

    if not suggestions:
        if result.final_score >= 0.85:
            suggestions.append("Strong match — apply with confidence.")
        else:
            suggestions.append(
                "Good match overall. Tailor your summary section to the specific role."
            )

    return suggestions


# ── Main scorer ───────────────────────────────────────────────────────────────


class MatchScorer:
    """
    Computes a 4-component match score between a resume and a job description.

    Weights are loaded from configs/config.yaml (scoring.weights).
    """

    def __init__(self) -> None:
        cfg = _load_config()
        w = cfg["scoring"]["weights"]
        self.w_skills = w["skills_match"]
        self.w_exp = w["experience_relevance"]
        self.w_edu = w["education_fit"]
        self.w_kw = w["keyword_alignment"]
        self._thresholds = cfg["scoring"]["thresholds"]
        logger.debug(
            f"Scorer weights — skills: {self.w_skills}, exp: {self.w_exp}, "
            f"edu: {self.w_edu}, kw: {self.w_kw}"
        )

    def _label(self, score_pct: float) -> str:
        t = self._thresholds
        if score_pct >= t["excellent"]:
            return "Excellent"
        elif score_pct >= t["good"]:
            return "Good"
        elif score_pct >= t["fair"]:
            return "Fair"
        return "Poor"

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
        Compute the full match score.

        Args:
            resume_text: Raw resume text.
            jd_text: Raw job description text.
            resume_skills: Pre-extracted skills (computed via NER if None).
            jd_skills: Pre-extracted skills (computed via NER if None).
            resume_emb: Pre-computed embedding (computed on the fly if None).
            jd_emb: Pre-computed embedding (computed on the fly if None).

        Returns:
            MatchResult with all scores and suggestions.
        """
        # Lazily compute embeddings if not provided
        if resume_emb is None or jd_emb is None:
            try:
                from src.features.embeddings import get_model

                model = get_model()
                if resume_emb is None:
                    resume_emb = model.encode(resume_text)
                if jd_emb is None:
                    jd_emb = model.encode(jd_text)
            except Exception as e:
                logger.warning(
                    f"Embedding unavailable, falling back to text-only scoring: {e}"
                )

        # Lazily extract skills if not provided
        if resume_skills is None or jd_skills is None:
            try:
                from src.features.ner import get_extractor

                extractor = get_extractor()
                if resume_skills is None:
                    resume_skills = set(extractor.extract(resume_text).technical_skills)
                if jd_skills is None:
                    jd_skills = set(extractor.extract(jd_text).technical_skills)
            except Exception as e:
                logger.warning(f"NER unavailable, using empty skill sets: {e}")
                resume_skills = set()
                jd_skills = set()

        # Component scores
        s_skills, matched, missing = score_skills_match(
            resume_skills, jd_skills, resume_emb, jd_emb
        )
        s_exp = score_experience_relevance(resume_text, jd_text, resume_emb, jd_emb)
        s_edu = score_education_fit(resume_text, jd_text)
        s_kw = score_keyword_alignment(resume_text, jd_text)

        # Weighted combination
        final = (
            self.w_skills * s_skills
            + self.w_exp * s_exp
            + self.w_edu * s_edu
            + self.w_kw * s_kw
        )
        final = float(np.clip(final, 0.0, 1.0))
        score_pct = round(final * 100, 1)

        result = MatchResult(
            final_score=final,
            score_pct=score_pct,
            label=self._label(score_pct),
            skills_match=s_skills,
            experience_relevance=s_exp,
            education_fit=s_edu,
            keyword_alignment=s_kw,
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
