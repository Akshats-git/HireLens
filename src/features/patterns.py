"""
Regex extraction of education level and years of experience.

Both the NER extractor and the match scorer need these, and previously kept
separate copies that had drifted apart. This module is the single definition.

Abbreviated degree names are only recognised in their dotted form ("B.E.",
"M.S.", "A.S.") because the bare two-letter forms collide with ordinary English:
an undotted `b\\.?e\\.?` pattern matches the word "be" and classifies almost any
resume as holding a bachelor's degree.
"""

import re

# Education levels ordered from lowest to highest. Index doubles as the rank
# used for gap scoring, so the order is significant.
DEGREE_ORDER: tuple[str, ...] = (
    "none",
    "bootcamp",
    "certification",
    "diploma",
    "associate",
    "bachelors",
    "masters",
    "phd",
)
DEGREE_RANK: dict[str, int] = {degree: i for i, degree in enumerate(DEGREE_ORDER)}

# Checked highest-first: the first match wins, so a resume listing both a
# master's and a bachelor's reports the master's.
_EDUCATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "phd",
        re.compile(r"\bph\.?\s?d\.?\b|\bdoctora(?:te|l)\b|\bd\.\s?phil\b", re.I),
    ),
    (
        "masters",
        re.compile(
            r"\bmaster(?:'?s)?\b|\bm\.\s?s\.?|\bm\.?sc\b|\bm\.\s?eng\b"
            r"|\bm\.?b\.?a\b|\bm\.?tech\b|\bpost\s?graduate\b",
            re.I,
        ),
    ),
    (
        "bachelors",
        re.compile(
            r"\bbachelor(?:'?s)?\b|\bb\.?s\.?\b|\bb\.?sc\b|\bb\.\s?e\.?"
            r"|\bb\.?eng\b|\bb\.?tech\b|\bb\.\s?a\.?|\bundergraduate\b",
            re.I,
        ),
    ),
    (
        "associate",
        re.compile(
            r"\bassociate(?:'?s)?\s+degree\b|\ba\.\s?a\.?\b|\ba\.\s?s\.?\s+degree\b",
            re.I,
        ),
    ),
    ("diploma", re.compile(r"\bdiploma\b", re.I)),
    (
        "certification",
        re.compile(r"\bcertificat(?:e|ion)s?\b|\bcertified\b", re.I),
    ),
    (
        "bootcamp",
        re.compile(r"\bboot\s?camp\b|\bcoding\s+school\b|\bnanodegree\b", re.I),
    ),
]

# Matches "5 years experience", "5+ years of experience", "3-5 years of work",
# "3 to 5 years experience". Captures the leading number.
EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d+\s*)?"
    r"years?\s*(?:of\s+)?(?:experience|exp\b|work)",
    re.I,
)


def detect_education_level(text: str, default: str = "none") -> str:
    """
    Return the highest education level mentioned in the text.

    Args:
        text: Resume or job description text.
        default: Value returned when no degree is mentioned.

    Returns:
        One of DEGREE_ORDER, or `default` if nothing matched.
    """
    for level, pattern in _EDUCATION_PATTERNS:
        if pattern.search(text):
            return level
    return default


def extract_experience_years(text: str) -> float:
    """
    Return the largest number of years of experience mentioned in the text.

    Returns 0.0 when the text does not state a duration.
    """
    matches = EXPERIENCE_PATTERN.findall(text)
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def required_experience_years(text: str) -> float | None:
    """
    Return the first duration mentioned in a job description — the requirement.

    Unlike `extract_experience_years`, the first match is used rather than the
    largest, because a posting states its requirement up front and may mention
    other durations later.
    """
    match = EXPERIENCE_PATTERN.search(text)
    return float(match.group(1)) if match else None
