"""
Preprocessing and training-pair construction.

Stages:
  1. Clean resume and job description text
  2. Split resumes into named sections
  3. Pair resumes with postings using category keywords as weak supervision
  4. Split into train / validation / test and write to data/processed/
"""

import os
import re
import unicodedata

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import PROJECT_ROOT, get_section
from src.logging_setup import configure_logging

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Section headers are short lines; longer lines are body text that merely
# happens to contain a header keyword.
MAX_HEADER_CHARS = 60

# A section needs more than this many characters to count as populated.
MIN_SECTION_CHARS = 20

PAIR_COLUMNS = (
    "resume_id",
    "resume_text",
    "job_description",
    "label",
    "resume_category",
    "pair_type",
)


def _mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _mlflow_experiment() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", get_section("mlflow")["experiment_name"])


# ── Patterns ──────────────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "education": re.compile(
        r"\b(education|academic|qualification|degree|university|college|school)\b", re.I
    ),
    "experience": re.compile(
        r"\b(experience|employment|work history|career|professional background)\b", re.I
    ),
    "skills": re.compile(
        r"\b(skills|technologies|tools|competencies|proficiencies|technical)\b", re.I
    ),
    "projects": re.compile(
        r"\b(projects|portfolio|achievements|accomplishments)\b", re.I
    ),
    "certifications": re.compile(
        r"\b(certifications?|certificates?|licenses?|credentials?)\b", re.I
    ),
    "summary": re.compile(r"\b(summary|objective|profile|about me|overview)\b", re.I),
}

# Unicode punctuation normalised to ASCII equivalents during cleaning.
_PUNCTUATION_REPLACEMENTS = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "•": "*",
    "·": "*",
    "…": "...",
    "﻿": "",
}

# Resume category to the posting keywords that mark a plausible match. Used as
# weak supervision: a posting containing any keyword is a positive for that
# category, and postings containing none are hard negatives.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Data Science": [
        "machine learning",
        "data scientist",
        "analytics",
        "python",
        "statistics",
    ],
    "HR": ["human resources", "recruiter", "talent acquisition", "hr manager"],
    "Advocate": ["lawyer", "attorney", "legal counsel", "litigation"],
    "Arts": ["graphic design", "creative", "artist", "visual design"],
    "Web Designing": ["frontend developer", "ui/ux", "web designer", "html", "css"],
    "Mechanical Engineer": [
        "mechanical engineer",
        "cad",
        "solidworks",
        "manufacturing",
    ],
    "Sales": ["sales manager", "business development", "account executive", "revenue"],
    "Health and fitness": [
        "fitness trainer",
        "nutritionist",
        "health coach",
        "wellness",
    ],
    "Civil Engineer": ["civil engineer", "structural", "construction", "autocad"],
    "Java Developer": ["java developer", "spring boot", "microservices", "backend"],
    "Business Analyst": [
        "business analyst",
        "requirements gathering",
        "stakeholders",
        "process",
    ],
    "SAP Developer": ["sap", "abap", "erp", "sap consultant"],
    "Automobile": ["automobile", "automotive engineer", "vehicle design"],
    "Agriculture": ["agriculture", "agronomist", "farming", "crop"],
    "Blockchain": ["blockchain", "ethereum", "solidity", "smart contracts", "web3"],
    "Testing": ["qa engineer", "test automation", "selenium", "quality assurance"],
    "Finance": ["financial analyst", "accounting", "cfa", "investment banking"],
    "Apparel": ["fashion designer", "textile", "apparel", "merchandiser"],
    "Digital Media": ["digital marketing", "seo", "content creator", "social media"],
    "DotNet Developer": [".net developer", "c#", "asp.net", "azure"],
    "Database": ["database administrator", "dba", "sql", "postgresql", "oracle"],
    "Electrical Engineering": [
        "electrical engineer",
        "power systems",
        "plc",
        "embedded",
    ],
    "Construction": ["construction manager", "project manager", "site engineer"],
    "Public Relations": ["pr manager", "communications", "media relations", "press"],
    "Operations Manager": [
        "operations manager",
        "supply chain",
        "logistics",
        "process improvement",
    ],
    "Python Developer": [
        "python developer",
        "django",
        "flask",
        "fastapi",
        "backend python",
    ],
    "ETL Developer": [
        "etl developer",
        "data pipeline",
        "airflow",
        "spark",
        "data engineer",
    ],
    "Network Security Engineer": [
        "network security",
        "cybersecurity",
        "firewall",
        "soc",
    ],
    "PMO": ["project management", "pmo", "pmp", "agile", "scrum master"],
    "Chef": ["chef", "culinary", "kitchen manager", "food"],
    "Consultant": ["consultant", "advisory", "strategy", "management consulting"],
    "BPO": ["bpo", "customer service", "call center", "support representative"],
    "DevOps Engineer": ["devops", "ci/cd", "kubernetes", "docker", "infrastructure"],
}


# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Normalise raw extracted text for embedding.

    Applies NFKC normalisation, strips control characters and contact details,
    collapses whitespace and repeated punctuation, and drops decorative rules.

    Args:
        text: Raw text.

    Returns:
        Cleaned text, empty if the input was blank.
    """
    if not text or not text.strip():
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )
    for source, replacement in _PUNCTUATION_REPLACEMENTS.items():
        text = text.replace(source, replacement)

    # Contact details carry no matching signal and leak PII into embeddings.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", " ", text)

    # Collapse runs such as "---" or "..." to a single character.
    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and not re.fullmatch(r"[\-_=*#|.]{2,}", line):
            lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ── Section detection ─────────────────────────────────────────────────────────


def detect_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections by header matching.

    A short line matching a header pattern opens a section; everything until the
    next header belongs to it. Text before any header lands in 'other'.

    Args:
        text: Cleaned resume text.

    Returns:
        Mapping of section name to its content.
    """
    sections: dict[str, list[str]] = {name: [] for name in _SECTION_PATTERNS}
    sections["other"] = []

    current = "other"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        header = None
        if len(stripped) < MAX_HEADER_CHARS:
            header = next(
                (
                    name
                    for name, pattern in _SECTION_PATTERNS.items()
                    if pattern.search(stripped)
                ),
                None,
            )

        if header:
            current = header
        else:
            sections[current].append(stripped)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def has_required_sections(sections: dict[str, str], min_sections: int = 2) -> bool:
    """True if at least `min_sections` detected sections have real content."""
    populated = sum(1 for text in sections.values() if len(text) > MIN_SECTION_CHARS)
    return populated >= min_sections


# ── Training pairs ────────────────────────────────────────────────────────────


def _index_jobs_by_category(
    categories: list[str], descriptions: list[str]
) -> dict[str, np.ndarray]:
    """
    Map each resume category to the indices of postings matching its keywords.

    Built once per category rather than once per resume. Scanning the full
    postings table inside the resume loop costs one pass per resume — tens of
    thousands of passes over a table with six figures of rows.

    Args:
        categories: Distinct resume categories.
        descriptions: Job descriptions, in DataFrame row order.

    Returns:
        Mapping of category to a sorted array of matching row indices.
    """
    lowered = [description.lower() for description in descriptions]

    matches: dict[str, np.ndarray] = {}
    for category in categories:
        keywords = _CATEGORY_KEYWORDS.get(category, [category.lower()])
        matches[category] = np.array(
            [
                index
                for index, description in enumerate(lowered)
                if any(keyword in description for keyword in keywords)
            ],
            dtype=int,
        )
        if matches[category].size == 0:
            logger.warning(f"No postings matched category '{category}'.")

    return matches


def build_training_pairs(
    resume_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    negatives_per_positive: int = 2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build labelled resume/posting pairs using category keywords as weak supervision.

    Positives pair a resume with a posting containing one of its category's
    keywords. Negatives pair it with postings matching none of them — a different
    professional domain rather than random noise.

    Args:
        resume_df: Resumes with columns [id, resume_text, category].
        jobs_df: Postings with a 'description' column.
        negatives_per_positive: Negative pairs generated per positive pair.
        random_seed: Seed for reproducible sampling.

    Returns:
        DataFrame with [resume_id, resume_text, job_description, label,
        resume_category, pair_type].
    """
    rng = np.random.default_rng(random_seed)
    logger.info("Building training pairs via category-based weak supervision...")

    descriptions = jobs_df["description"].tolist()
    positive_index = _index_jobs_by_category(
        resume_df["category"].unique().tolist(), descriptions
    )
    all_indices = np.arange(len(descriptions))
    negative_index = {
        category: np.setdiff1d(all_indices, positives, assume_unique=False)
        for category, positives in positive_index.items()
    }

    pairs: list[dict] = []
    for row in tqdm(
        resume_df.itertuples(index=False), total=len(resume_df), desc="Pairing resumes"
    ):
        positives = positive_index[row.category]
        if positives.size == 0:
            continue

        def add_pair(job_index: int, label: float, pair_type: str) -> None:
            pairs.append(
                {
                    "resume_id": row.id,
                    "resume_text": row.resume_text,
                    "job_description": descriptions[job_index],
                    "label": label,
                    "resume_category": row.category,
                    "pair_type": pair_type,
                }
            )

        add_pair(int(rng.choice(positives)), 1.0, "positive")

        negatives = negative_index[row.category]
        if negatives.size == 0:
            continue
        sample_size = min(negatives_per_positive, negatives.size)
        for job_index in rng.choice(negatives, size=sample_size, replace=False):
            add_pair(int(job_index), 0.0, "negative")

    # Named columns are declared explicitly so that an empty result — every
    # category unmatched — is still a well-formed frame rather than one with no
    # columns at all, which would fail on the label lookups below and downstream.
    pairs_df = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    if pairs_df.empty:
        logger.warning(
            "No training pairs were built: no posting matched any resume category. "
            "Check the category keywords against the postings corpus."
        )
        return pairs_df

    logger.info(
        f"Built {len(pairs_df)} pairs: "
        f"{(pairs_df['label'] == 1.0).sum()} positive, "
        f"{(pairs_df['label'] == 0.0).sum()} negative."
    )
    return pairs_df


def split_pairs(
    pairs_df: pd.DataFrame, split: list[float], random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split pairs into train / validation / test, stratified on the label.

    Args:
        pairs_df: All labelled pairs.
        split: Three fractions summing to 1.0.
        random_seed: Seed for reproducible splits.

    Returns:
        (train_df, val_df, test_df)
    """
    train_fraction, val_fraction, test_fraction = split
    holdout_fraction = val_fraction + test_fraction

    train_df, holdout_df = train_test_split(
        pairs_df,
        test_size=holdout_fraction,
        random_state=random_seed,
        stratify=pairs_df["label"],
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_fraction / holdout_fraction,
        random_state=random_seed,
        stratify=holdout_df["label"],
    )
    logger.info(
        f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ── Entry point ───────────────────────────────────────────────────────────────


def run_preprocessing(
    resume_df: pd.DataFrame | None = None,
    jobs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Clean both datasets, build labelled pairs, split them, and write to disk.

    Args:
        resume_df: Pre-loaded resumes; read from data/raw/ if omitted.
        jobs_df: Pre-loaded postings; read from data/raw/ if omitted.

    Returns:
        The full pairs DataFrame, before splitting.
    """
    logger.info("=" * 60)
    logger.info("HireLens preprocessing")
    logger.info("=" * 60)

    data_config = get_section("data")
    settings = data_config["preprocessing"]
    random_seed = data_config["random_seed"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if resume_df is None:
        resume_df = _read_required(RAW_DIR / "resumes_clean.csv", "resumes")
    if jobs_df is None:
        jobs_df = _read_required(RAW_DIR / "jobs_clean.csv", "job postings")

    logger.info("Cleaning resume text...")
    resume_df = resume_df.copy()
    resume_df["resume_text_clean"] = resume_df["resume_text"].apply(clean_text)
    before = len(resume_df)
    resume_df = resume_df[
        resume_df["resume_text_clean"].str.len() > settings["min_resume_chars"]
    ].reset_index(drop=True)
    logger.info(f"After cleaning: {before} → {len(resume_df)} resumes")

    logger.info("Detecting resume sections...")
    sections_df = pd.DataFrame(
        resume_df["resume_text_clean"].apply(detect_sections).tolist()
    )
    resume_df = pd.concat([resume_df, sections_df], axis=1)

    logger.info("Cleaning job descriptions...")
    jobs_df = jobs_df.copy()
    jobs_df["description"] = jobs_df["description"].apply(clean_text)
    jobs_df = jobs_df[
        jobs_df["description"].str.len() > settings["min_jd_chars"]
    ].reset_index(drop=True)

    pairs_df = build_training_pairs(
        resume_df,
        jobs_df,
        negatives_per_positive=settings["negatives_per_positive"],
        random_seed=random_seed,
    )
    train_df, val_df, test_df = split_pairs(
        pairs_df, data_config["train_val_test_split"], random_seed
    )

    outputs = {
        "resumes_processed.csv": resume_df,
        "jobs_processed.csv": jobs_df,
        "train_pairs.csv": train_df,
        "val_pairs.csv": val_df,
        "test_pairs.csv": test_df,
    }
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / name, index=False)
    logger.info(f"Wrote {len(outputs)} files to {PROCESSED_DIR}")

    _log_to_mlflow(resume_df, jobs_df, pairs_df, train_df, val_df, test_df)

    logger.success("Preprocessing complete.")
    return pairs_df


def _read_required(path, label: str) -> pd.DataFrame:
    """Read a CSV the pipeline depends on, with an actionable error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label.capitalize()} CSV not found at {path}. Run the ingest stage first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} {label} from {path}")
    return df


def _log_to_mlflow(resume_df, jobs_df, pairs_df, train_df, val_df, test_df) -> None:
    """Record preprocessing counts to MLflow. Failures are non-fatal."""
    try:
        mlflow.set_tracking_uri(_mlflow_tracking_uri())
        mlflow.set_experiment(_mlflow_experiment())
        with mlflow.start_run(run_name="data-preprocessing"):
            mlflow.log_metrics(
                {
                    "total_resumes_processed": float(len(resume_df)),
                    "total_jobs_processed": float(len(jobs_df)),
                    "total_pairs": float(len(pairs_df)),
                    "train_pairs": float(len(train_df)),
                    "val_pairs": float(len(val_df)),
                    "test_pairs": float(len(test_df)),
                    "positive_pair_ratio": float((pairs_df["label"] == 1.0).mean()),
                }
            )
            mlflow.set_tag("stage", "preprocessing")
            logger.success("Preprocessing statistics logged to MLflow.")
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    configure_logging(log_file="preprocessing.log")
    run_preprocessing()
