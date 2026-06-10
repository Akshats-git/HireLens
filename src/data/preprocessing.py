"""
Preprocessing pipeline for HireLens.

Stages:
  1. PDF text extraction via pdfplumber
  2. Text cleaning (noise removal, unicode normalisation, whitespace)
  3. Section detection (Education, Experience, Skills, Projects)
  4. Weak-supervision pair construction via category matching
  5. Save processed data to data/processed/
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pdfplumber
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "hirelens-resume-matching")

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)
logger.add(
    PROJECT_ROOT / "logs" / "preprocessing.log",
    rotation="50 MB",
    retention="14 days",
    level="DEBUG",
)


# ── Section patterns ──────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "education": re.compile(
        r"\b(education|academic|qualification|degree|university|college|school)\b",
        re.IGNORECASE,
    ),
    "experience": re.compile(
        r"\b(experience|employment|work history|career|professional background|job)\b",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"\b(skills|technologies|tools|competencies|proficiencies|technical)\b",
        re.IGNORECASE,
    ),
    "projects": re.compile(
        r"\b(projects|portfolio|achievements|accomplishments)\b",
        re.IGNORECASE,
    ),
    "certifications": re.compile(
        r"\b(certifications?|certificates?|licenses?|credentials?)\b",
        re.IGNORECASE,
    ),
    "summary": re.compile(
        r"\b(summary|objective|profile|about me|overview)\b",
        re.IGNORECASE,
    ),
}

# Maps resume category labels → job description keywords for weak supervision
_CATEGORY_TO_JD_KEYWORDS: dict[str, list[str]] = {
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


# ── PDF extraction ────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using pdfplumber.

    Falls back to an empty string on any error rather than crashing the pipeline.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text)
            return "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF extraction failed for {pdf_path}: {e}")
        return ""


def extract_texts_from_pdf_dir(pdf_dir: Path) -> list[dict[str, str]]:
    """
    Batch-extract text from all PDFs in a directory.

    Args:
        pdf_dir: Directory containing .pdf files.

    Returns:
        List of dicts with keys 'filename' and 'text'.
    """
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    results = []
    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            results.append({"filename": pdf_path.name, "text": text})
        else:
            logger.debug(f"Empty extraction: {pdf_path.name}")

    logger.info(f"Successfully extracted {len(results)}/{len(pdf_files)} PDFs.")
    return results


# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline for resume / job description text.

    Steps:
        1. Unicode normalisation (NFKC)
        2. Remove non-printable / control characters
        3. Replace common unicode punctuation with ASCII equivalents
        4. Collapse repeated punctuation and whitespace
        5. Strip leading/trailing whitespace per line
        6. Remove lines that are purely decorative (dashes, dots, stars)

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove control characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )

    # 3. Replace unicode punctuation variants
    replacements = {
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
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # 4. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 6. Remove phone numbers
    text = re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", " ", text)

    # 7. Remove repeated punctuation (e.g. "---", "***", "...")
    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)

    # 8. Collapse whitespace within lines, strip each line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        # Drop decorative separator lines
        if re.fullmatch(r"[\-_=*#|.]{2,}", line):
            continue
        if line:
            lines.append(line)

    # 9. Collapse 3+ consecutive blank lines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


# ── Section detection ─────────────────────────────────────────────────────────


def detect_sections(text: str) -> dict[str, str]:
    """
    Detect and extract named sections from resume text using header matching.

    Uses a line-by-line scan: when a line matches a section header pattern,
    all subsequent lines until the next header are assigned to that section.

    Args:
        text: Cleaned resume text.

    Returns:
        Dict mapping section name → section text content.
    """
    sections: dict[str, list[str]] = {name: [] for name in _SECTION_PATTERNS}
    sections["other"] = []

    current_section = "other"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line is a section header (short line matching a pattern)
        matched_section = None
        if len(stripped) < 60:
            for section_name, pattern in _SECTION_PATTERNS.items():
                if pattern.search(stripped):
                    matched_section = section_name
                    break

        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(stripped)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def has_required_sections(sections: dict[str, str], min_sections: int = 2) -> bool:
    """Return True if the resume has at least min_sections non-empty detected sections."""
    non_empty = sum(1 for v in sections.values() if len(v) > 20)
    return non_empty >= min_sections


# ── Training pair construction ────────────────────────────────────────────────


def build_training_pairs(
    resume_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    neg_ratio: int = 2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Construct positive and hard-negative training pairs via weak supervision.

    Strategy:
      - POSITIVE: Resume category matched to a job description containing
        category-specific keywords (category matching as weak supervision).
      - NEGATIVE: Resume paired with a job from a different category bucket
        (hard negatives — different domain, not random noise).

    Args:
        resume_df: DataFrame with columns [id, resume_text, category].
        jobs_df: DataFrame with columns [job_id, title, description, ...].
        neg_ratio: Number of negative pairs per positive pair.
        random_seed: For reproducibility.

    Returns:
        DataFrame with columns:
          [resume_id, resume_text, job_description, label, resume_category, pair_type]
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)

    logger.info("Building training pairs via category-based weak supervision...")

    pairs: list[dict[str, Any]] = []
    jobs_list = jobs_df.to_dict("records")

    for _, resume_row in tqdm(
        resume_df.iterrows(), total=len(resume_df), desc="Pairing resumes"
    ):
        category = resume_row["category"]
        keywords = _CATEGORY_TO_JD_KEYWORDS.get(category, [category.lower()])

        # Find positive jobs: descriptions containing category keywords
        positive_jobs = [
            j
            for j in jobs_list
            if any(kw in j["description"].lower() for kw in keywords)
        ]

        if not positive_jobs:
            continue

        # Sample 1 positive pair
        pos_job = positive_jobs[rng.integers(len(positive_jobs))]
        pairs.append(
            {
                "resume_id": resume_row["id"],
                "resume_text": resume_row["resume_text"],
                "job_description": pos_job["description"],
                "label": 1.0,
                "resume_category": category,
                "pair_type": "positive",
            }
        )

        # Sample hard negatives: jobs that do NOT match this category
        negative_jobs = [
            j
            for j in jobs_list
            if not any(kw in j["description"].lower() for kw in keywords)
        ]

        if not negative_jobs:
            continue

        n_neg = min(neg_ratio, len(negative_jobs))
        neg_indices = rng.choice(len(negative_jobs), size=n_neg, replace=False)
        for idx in neg_indices:
            pairs.append(
                {
                    "resume_id": resume_row["id"],
                    "resume_text": resume_row["resume_text"],
                    "job_description": negative_jobs[idx]["description"],
                    "label": 0.0,
                    "resume_category": category,
                    "pair_type": "negative",
                }
            )

    pairs_df = pd.DataFrame(pairs)
    pos_count = (pairs_df["label"] == 1.0).sum()
    neg_count = (pairs_df["label"] == 0.0).sum()
    logger.info(
        f"Built {len(pairs_df)} pairs: {pos_count} positive, {neg_count} negative."
    )
    return pairs_df


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_preprocessing(
    resume_df: pd.DataFrame | None = None,
    jobs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: clean → detect sections → build pairs → save.

    Args:
        resume_df: Pre-loaded resume DataFrame (loaded from CSV if None).
        jobs_df: Pre-loaded jobs DataFrame (loaded from CSV if None).

    Returns:
        Training pairs DataFrame.
    """
    from src.data.ingestion import RAW_DIR

    logger.info("=" * 60)
    logger.info("Starting HireLens preprocessing pipeline")
    logger.info("=" * 60)

    # Load if not provided
    if resume_df is None:
        resume_csv = RAW_DIR / "resumes_clean.csv"
        if not resume_csv.exists():
            raise FileNotFoundError(
                f"Resume CSV not found at {resume_csv}. Run ingestion first."
            )
        resume_df = pd.read_csv(resume_csv)
        logger.info(f"Loaded {len(resume_df)} resumes from {resume_csv}")

    if jobs_df is None:
        jobs_csv = RAW_DIR / "jobs_clean.csv"
        if not jobs_csv.exists():
            raise FileNotFoundError(
                f"Jobs CSV not found at {jobs_csv}. Run ingestion first."
            )
        jobs_df = pd.read_csv(jobs_csv)
        logger.info(f"Loaded {len(jobs_df)} jobs from {jobs_csv}")

    # ── Step 1: Clean resume text ─────────────────────────────────────────────
    logger.info("Cleaning resume text...")
    resume_df = resume_df.copy()
    resume_df["resume_text_clean"] = resume_df["resume_text"].apply(clean_text)

    # Drop too-short cleaned texts
    before = len(resume_df)
    resume_df = resume_df[resume_df["resume_text_clean"].str.len() > 100]
    logger.info(f"After cleaning: {before} → {len(resume_df)} resumes")

    # ── Step 2: Detect sections ───────────────────────────────────────────────
    logger.info("Detecting resume sections...")
    section_records = resume_df["resume_text_clean"].apply(detect_sections)
    sections_df = pd.DataFrame(section_records.tolist())
    resume_df = pd.concat([resume_df.reset_index(drop=True), sections_df], axis=1)

    # ── Step 3: Clean job descriptions ───────────────────────────────────────
    logger.info("Cleaning job descriptions...")
    jobs_df = jobs_df.copy()
    jobs_df["description"] = jobs_df["description"].apply(clean_text)
    jobs_df = jobs_df[jobs_df["description"].str.len() > 50]

    # ── Step 4: Build training pairs ──────────────────────────────────────────
    pairs_df = build_training_pairs(resume_df, jobs_df)

    # ── Step 5: Train/val/test split ──────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        pairs_df, test_size=0.20, random_state=42, stratify=pairs_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    logger.info(
        f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    resume_out = PROCESSED_DIR / "resumes_processed.csv"
    jobs_out = PROCESSED_DIR / "jobs_processed.csv"
    train_out = PROCESSED_DIR / "train_pairs.csv"
    val_out = PROCESSED_DIR / "val_pairs.csv"
    test_out = PROCESSED_DIR / "test_pairs.csv"

    resume_df.to_csv(resume_out, index=False)
    jobs_df.to_csv(jobs_out, index=False)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"Saved processed files to {PROCESSED_DIR}")

    # ── Step 7: Log to MLflow ─────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
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
            logger.success("Preprocessing stats logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.success("Preprocessing pipeline complete.")
    return pairs_df


if __name__ == "__main__":
    run_preprocessing()
