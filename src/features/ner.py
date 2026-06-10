"""
NER and skill extraction module for HireLens.

Extracts from resume / job description text:
  - Technical skills    (matched against skills_taxonomy.json)
  - Soft skills
  - Years of experience
  - Education level
  - Job titles

Uses spaCy en_core_web_trf (falls back to en_core_web_sm if trf not installed).
Taxonomy matching uses Aho-Corasick via the `pyahocorasick` library for O(n) throughput.
Falls back to set-intersection if pyahocorasick is not installed.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import spacy
import yaml
from loguru import logger
from spacy.language import Language

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
_TAXONOMY_PATH = PROJECT_ROOT / "data" / "skills_taxonomy.json"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Structured output from NER extraction."""

    technical_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = "unknown"
    job_titles: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    raw_entities: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "technical_skills": self.technical_skills,
            "soft_skills": self.soft_skills,
            "experience_years": self.experience_years,
            "education_level": self.education_level,
            "job_titles": self.job_titles,
            "certifications": self.certifications,
            "raw_entities": self.raw_entities,
        }

    @property
    def all_skills(self) -> list[str]:
        return self.technical_skills + self.soft_skills


# ── Education patterns ────────────────────────────────────────────────────────

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
    ("associate", re.compile(r"\b(associate(?:\'?s)?|a\.?s\.?|a\.?a\.?)\b", re.I)),
    ("diploma", re.compile(r"\b(diploma|certificate program)\b", re.I)),
    ("bootcamp", re.compile(r"\b(bootcamp|coding school|nanodegree)\b", re.I)),
]

# Years of experience patterns: "5 years", "5+ years", "3-5 years", "over 10 years"
_EXP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s+)?(?:experience|exp\.?|work)",
    re.I,
)

# Job title indicators
_TITLE_CONTEXT = re.compile(
    r"(?:as|title|position|role|work(?:ed|ing)?\s+as|currently|previously|senior|junior|lead)\s+"
    r"([A-Z][a-zA-Z\s]{3,40}?)(?:\sat\s|\sin\s|\s[-–]\s|\.|,|\n)",
)


# ── Taxonomy loader ───────────────────────────────────────────────────────────


def _load_taxonomy() -> tuple[set[str], set[str], set[str]]:
    """
    Load the skills taxonomy into three flat sets:
      (technical_skills, soft_skills, certifications)
    """
    with open(_TAXONOMY_PATH) as f:
        tax = json.load(f)

    technical: set[str] = set()
    for cat_skills in tax.get("technical", {}).values():
        technical.update(s.lower() for s in cat_skills)
    for cat_skills in tax.get("domain", {}).values():
        technical.update(s.lower() for s in cat_skills)

    soft: set[str] = {s.lower() for s in tax.get("soft", [])}
    certs: set[str] = {s.lower() for s in tax.get("certifications", [])}

    return technical, soft, certs


# ── Aho-Corasick multi-string search (optional speedup) ──────────────────────


def _build_automaton(skills: set[str]) -> Any | None:
    """Build an Aho-Corasick automaton if the library is available."""
    try:
        import ahocorasick

        A = ahocorasick.Automaton()
        for skill in skills:
            A.add_word(skill, skill)
        A.make_automaton()
        return A
    except ImportError:
        return None


def _search_with_automaton(text: str, automaton: Any) -> list[str]:
    found = []
    text_lower = text.lower()
    for _, skill in automaton.iter(text_lower):
        # Ensure whole-word match
        start = text_lower.find(skill)
        if start == -1:
            continue
        end = start + len(skill)
        before = text_lower[start - 1] if start > 0 else " "
        after = text_lower[end] if end < len(text_lower) else " "
        if not (before.isalnum() or before == "-") and not (
            after.isalnum() or after == "-"
        ):
            found.append(skill)
    return found


def _search_with_set(text: str, skills: set[str]) -> list[str]:
    """Fallback: naive substring search when Aho-Corasick is unavailable."""
    text_lower = text.lower()
    found = []
    for skill in skills:
        # Add word-boundary check for short skills to avoid false positives
        if len(skill) <= 3:
            if re.search(rf"\b{re.escape(skill)}\b", text_lower):
                found.append(skill)
        elif skill in text_lower:
            found.append(skill)
    return found


# ── spaCy loader ──────────────────────────────────────────────────────────────


def _load_spacy(config: dict) -> Language:
    """Load spaCy model with transformer fallback to sm."""
    trf_model = config["model"]["ner"]["model"]
    fallback = config["model"]["ner"]["fallback_model"]

    try:
        nlp = spacy.load(trf_model)
        logger.info(f"Loaded spaCy model: {trf_model}")
        return nlp
    except OSError:
        logger.warning(f"{trf_model} not found, trying {fallback}")
        try:
            nlp = spacy.load(fallback)
            logger.info(f"Loaded spaCy model: {fallback}")
            return nlp
        except OSError:
            logger.warning("No spaCy model found — NER entity extraction disabled.")
            return spacy.blank("en")


# ── Main extractor ────────────────────────────────────────────────────────────


class NERExtractor:
    """
    Extracts structured information from resume / JD text.

    Combines:
      - Taxonomy-based skill matching (fast, curated)
      - spaCy NER for named entities (persons, orgs, titles)
      - Regex patterns for education level and experience years
    """

    def __init__(self) -> None:
        with open(_CONFIG_PATH) as f:
            self._config = yaml.safe_load(f)

        self._technical, self._soft, self._certs = _load_taxonomy()
        logger.info(
            f"Taxonomy loaded: {len(self._technical)} technical, "
            f"{len(self._soft)} soft, {len(self._certs)} cert skills"
        )

        # Try to build Aho-Corasick automata for O(n) search
        self._tech_automaton = _build_automaton(self._technical)
        self._soft_automaton = _build_automaton(self._soft)
        self._cert_automaton = _build_automaton(self._certs)
        if self._tech_automaton:
            logger.debug("Using Aho-Corasick for skill matching.")
        else:
            logger.debug(
                "Using naive skill matching (pip install pyahocorasick for speedup)."
            )

        self._nlp = _load_spacy(self._config)

    def _match_skills(
        self, text: str, taxonomy: set[str], automaton: Any | None
    ) -> list[str]:
        if automaton:
            return _search_with_automaton(text, automaton)
        return _search_with_set(text, taxonomy)

    def _extract_education(self, text: str) -> str:
        """Return the highest detected education level."""
        for level, pattern in _EDU_PATTERNS:
            if pattern.search(text):
                return level
        return "unknown"

    def _extract_experience_years(self, text: str) -> float:
        """Return the maximum years of experience mentioned."""
        matches = _EXP_PATTERN.findall(text)
        if not matches:
            return 0.0
        return max(float(m) for m in matches)

    def _extract_job_titles(self, doc: Any) -> list[str]:
        """Extract job titles using spaCy NER + context patterns."""
        titles = []

        # spaCy entity-based (WORK_OF_ART, JOB are not standard, use context)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG"):
                continue  # skip non-title entities
            if ent.label_ in ("WORK_OF_ART",) and len(ent.text.split()) <= 5:
                titles.append(ent.text.strip())

        # Regex context patterns
        for match in _TITLE_CONTEXT.finditer(doc.text):
            title = match.group(1).strip()
            if 2 <= len(title.split()) <= 6:
                titles.append(title)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique = []
        for t in titles:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)
        return unique[:10]

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all structured fields from a single document.

        Args:
            text: Raw resume or job description text.

        Returns:
            ExtractionResult with all extracted fields.
        """
        if not text or not text.strip():
            return ExtractionResult()

        # Run spaCy pipeline
        doc = self._nlp(text[:100_000])  # cap to avoid OOM on huge texts

        # Skills
        technical = self._match_skills(text, self._technical, self._tech_automaton)
        soft = self._match_skills(text, self._soft, self._soft_automaton)
        certs = self._match_skills(text, self._certs, self._cert_automaton)

        # Raw NER entities
        raw_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        return ExtractionResult(
            technical_skills=sorted(set(technical)),
            soft_skills=sorted(set(soft)),
            experience_years=self._extract_experience_years(text),
            education_level=self._extract_education(text),
            job_titles=self._extract_job_titles(doc),
            certifications=sorted(set(certs)),
            raw_entities=raw_entities,
        )

    def extract_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[ExtractionResult]:
        """
        Extract from a list of texts using spaCy's pipe() for efficiency.

        Args:
            texts: List of text strings.
            batch_size: spaCy pipe batch size.

        Returns:
            List of ExtractionResult objects.
        """
        from tqdm import tqdm

        results = []
        docs = list(
            self._nlp.pipe(
                [t[:100_000] for t in texts],
                batch_size=batch_size,
            )
        )

        for text, doc in tqdm(
            zip(texts, docs), total=len(texts), desc="NER extraction"
        ):
            technical = self._match_skills(text, self._technical, self._tech_automaton)
            soft = self._match_skills(text, self._soft, self._soft_automaton)
            certs = self._match_skills(text, self._certs, self._cert_automaton)
            raw_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

            results.append(
                ExtractionResult(
                    technical_skills=sorted(set(technical)),
                    soft_skills=sorted(set(soft)),
                    experience_years=self._extract_experience_years(text),
                    education_level=self._extract_education(text),
                    job_titles=self._extract_job_titles(doc),
                    certifications=sorted(set(certs)),
                    raw_entities=raw_entities,
                )
            )

        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_default_extractor: NERExtractor | None = None


def get_extractor() -> NERExtractor:
    """Return the shared NERExtractor (loads on first call)."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = NERExtractor()
    return _default_extractor
