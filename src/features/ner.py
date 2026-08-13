"""
Skill and entity extraction for HireLens.

Extracts from resume / job description text:
  - Technical skills    (matched against data/skills_taxonomy.json)
  - Soft skills and certifications
  - Years of experience and education level (see src/features/patterns.py)
  - Job titles

Skill matching runs an Aho-Corasick automaton over the text for O(n) throughput,
falling back to per-skill regex search when pyahocorasick is unavailable.
Named entities come from spaCy's en_core_web_trf, falling back to en_core_web_sm
and finally to a blank pipeline if no model is installed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger
from spacy.language import Language

from src.config import PROJECT_ROOT, get_section
from src.features.patterns import detect_education_level, extract_experience_years

TAXONOMY_PATH = PROJECT_ROOT / "data" / "skills_taxonomy.json"

# spaCy allocates roughly linearly in document length; cap absurd inputs.
MAX_DOC_CHARS = 100_000

# Job titles rarely run past six words; anything longer is a sentence fragment.
MAX_TITLE_WORDS = 6
MAX_TITLES = 10

# Captures a capitalised phrase introduced by a role-indicating word.
_TITLE_CONTEXT = re.compile(
    r"(?:as|title|position|role|work(?:ed|ing)?\s+as|currently|previously|senior|junior|lead)\s+"
    r"([A-Z][a-zA-Z\s]{3,40}?)(?:\sat\s|\sin\s|\s[-–]\s|\.|,|\n)",
)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Structured fields pulled out of a single document."""

    technical_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = "none"
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


# ── Taxonomy ──────────────────────────────────────────────────────────────────


def load_taxonomy() -> tuple[set[str], set[str], set[str]]:
    """
    Load data/skills_taxonomy.json into flat lowercase sets.

    Returns:
        (technical_skills, soft_skills, certifications)
    """
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    technical: set[str] = set()
    for group in ("technical", "domain"):
        for skills in taxonomy.get(group, {}).values():
            technical.update(skill.lower() for skill in skills)

    soft = {skill.lower() for skill in taxonomy.get("soft", [])}
    certifications = {skill.lower() for skill in taxonomy.get("certifications", [])}

    return technical, soft, certifications


# ── Skill matching ────────────────────────────────────────────────────────────


def _build_automaton(skills: set[str]) -> Any | None:
    """Build an Aho-Corasick automaton, or return None if the library is missing."""
    if not skills:
        return None
    try:
        import ahocorasick
    except ImportError:
        return None

    automaton = ahocorasick.Automaton()
    for skill in skills:
        automaton.add_word(skill, skill)
    automaton.make_automaton()
    return automaton


def _is_whole_word(text: str, start: int, end: int) -> bool:
    """
    True if text[start:end] is not embedded in a larger token.

    A hyphen counts as part of the surrounding token so that searching for
    'react' does not match inside 'react-native', which is its own skill.
    """
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "-") and not (
        after.isalnum() or after == "-"
    )


def _search_with_automaton(text: str, automaton: Any) -> set[str]:
    """
    Find every whole-word taxonomy skill in the text.

    `Automaton.iter` yields the end offset of each hit, which is what the
    boundary check must use — resolving the skill's position with `str.find`
    instead would test the first occurrence in the document rather than this one.
    """
    lowered = text.lower()
    found: set[str] = set()
    for end_index, skill in automaton.iter(lowered):
        start = end_index - len(skill) + 1
        if _is_whole_word(lowered, start, end_index + 1):
            found.add(skill)
    return found


def _search_with_regex(text: str, skills: Iterable[str]) -> set[str]:
    """
    Fallback matcher used when pyahocorasick is not installed.

    Every skill is word-bounded, not just short ones: a plain substring test
    matches 'java' inside 'javascript' and 'go' inside 'algorithm'.
    """
    lowered = text.lower()
    return {
        skill
        for skill in skills
        if re.search(rf"(?<![\w-]){re.escape(skill)}(?![\w-])", lowered)
    }


# ── spaCy ─────────────────────────────────────────────────────────────────────


def _load_spacy(ner_config: dict) -> Language:
    """Load the configured spaCy model, degrading to the fallback then to blank."""
    import spacy

    for model_name in (ner_config["model"], ner_config["fallback_model"]):
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not installed.")

    logger.warning(
        "No spaCy model available — entity extraction disabled. "
        "Run scripts/install_models.sh to install them."
    )
    return spacy.blank("en")


# ── Extractor ─────────────────────────────────────────────────────────────────


class NERExtractor:
    """
    Extracts structured information from resume / job description text.

    Combines curated taxonomy matching for skills, spaCy NER for named entities,
    and regex patterns for education level and years of experience.
    """

    def __init__(self) -> None:
        ner_config = get_section("model")["ner"]
        self._batch_size = ner_config.get("batch_size", 128)

        self._technical, self._soft, self._certifications = load_taxonomy()
        logger.info(
            f"Taxonomy loaded: {len(self._technical)} technical, "
            f"{len(self._soft)} soft, {len(self._certifications)} certification skills"
        )

        self._automatons = {
            "technical": _build_automaton(self._technical),
            "soft": _build_automaton(self._soft),
            "certifications": _build_automaton(self._certifications),
        }
        if self._automatons["technical"] is None:
            logger.warning(
                "pyahocorasick is not installed — falling back to regex skill "
                "matching, which is markedly slower on bulk uploads."
            )

        self._nlp = _load_spacy(ner_config)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _match_skills(self, text: str, group: str, taxonomy: set[str]) -> list[str]:
        automaton = self._automatons[group]
        found = (
            _search_with_automaton(text, automaton)
            if automaton is not None
            else _search_with_regex(text, taxonomy)
        )
        return sorted(found)

    def _extract_job_titles(self, doc: Any) -> list[str]:
        """Collect job titles from spaCy entities and role-context regex matches."""
        titles: list[str] = []

        for ent in doc.ents:
            if ent.label_ == "WORK_OF_ART" and len(ent.text.split()) <= MAX_TITLE_WORDS:
                titles.append(ent.text.strip())

        for match in _TITLE_CONTEXT.finditer(doc.text):
            title = match.group(1).strip()
            if 2 <= len(title.split()) <= MAX_TITLE_WORDS:
                titles.append(title)

        seen: set[str] = set()
        unique: list[str] = []
        for title in titles:
            key = title.lower()
            if key not in seen:
                seen.add(key)
                unique.append(title)
        return unique[:MAX_TITLES]

    def _build_result(self, text: str, doc: Any) -> ExtractionResult:
        return ExtractionResult(
            technical_skills=self._match_skills(text, "technical", self._technical),
            soft_skills=self._match_skills(text, "soft", self._soft),
            certifications=self._match_skills(
                text, "certifications", self._certifications
            ),
            experience_years=extract_experience_years(text),
            education_level=detect_education_level(text),
            job_titles=self._extract_job_titles(doc),
            raw_entities=[{"text": ent.text, "label": ent.label_} for ent in doc.ents],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all structured fields from a single document.

        Args:
            text: Raw resume or job description text.

        Returns:
            ExtractionResult; empty if the text is blank.
        """
        if not text or not text.strip():
            return ExtractionResult()

        return self._build_result(text, self._nlp(text[:MAX_DOC_CHARS]))

    def extract_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[ExtractionResult]:
        """
        Extract from many documents, sharing one spaCy pipeline pass.

        Args:
            texts: Raw document texts.
            batch_size: spaCy pipe batch size. Falls back to the configured value.

        Returns:
            One ExtractionResult per input, in the same order.
        """
        truncated = [text[:MAX_DOC_CHARS] for text in texts]
        docs = self._nlp.pipe(truncated, batch_size=batch_size or self._batch_size)
        return [self._build_result(text, doc) for text, doc in zip(texts, docs)]


# ── Module-level singleton ────────────────────────────────────────────────────

_default_extractor: NERExtractor | None = None


def get_extractor() -> NERExtractor:
    """Return the shared NERExtractor, loading models on first call."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = NERExtractor()
    return _default_extractor
