"""
Held-out evaluation set construction.

Nothing here is synthetic: the pairs come from a structured Kaggle resume
dataset joined against real job postings. Each structured resume row is rendered
as resume text, paired with sampled postings, and scored by the same four rules
the production scorer uses — skill overlap, experience fit, education fit, and
title/keyword match — to give a reference score independent of the embedding model.

Output: data/eval/eval_pairs.csv
Columns: resume_text, job_description, match_score, category, score_breakdown
"""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import PROJECT_ROOT, get_section
from src.features.patterns import (
    DEGREE_RANK,
    detect_education_level,
    required_experience_years,
)
from src.logging_setup import configure_logging

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_PAIRS_PATH = EVAL_DIR / "eval_pairs.csv"

EVAL_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "eval_dataset" / "resume_dataset_1200.csv"
)
JOBS_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_clean.csv"

# Component weights, mirroring scoring.weights in configs/config.yaml.
_WEIGHTS = {
    "skills_match": 0.40,
    "experience_relevance": 0.30,
    "education_fit": 0.15,
    "keyword_alignment": 0.15,
}

# Scores used when a field gives no signal either way.
_NEUTRAL_SKILLS = 0.3
_NEUTRAL_EXPERIENCE = 0.6

# Postings shorter than this are stubs rather than real descriptions.
MIN_JD_CHARS = 100

_MISSING_VALUES = frozenset({"", "nan", "none"})


def _mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _mlflow_experiment() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", get_section("mlflow")["experiment_name"])


def _clean_field(value: Any) -> str:
    """Return a stripped string, or '' for pandas' many spellings of missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_VALUES else text


# ── Resume rendering ──────────────────────────────────────────────────────────


def build_resume_text(row: pd.Series) -> str:
    """
    Render a structured resume row as plain resume text.

    Args:
        row: A row of the structured Kaggle resume dataset.

    Returns:
        Formatted resume text suitable for embedding.
    """
    parts: list[str] = [_clean_field(row.get("Name")) or "Candidate", ""]

    title = _clean_field(row.get("Current_Job_Title"))
    years = _clean_field(row.get("Experience_Years")) or "0"
    parts.append("PROFESSIONAL SUMMARY")
    parts.append(
        f"Experienced {title} with {years} year(s) of professional experience."
        if title
        else f"Recent graduate with {years} year(s) of experience seeking new opportunities."
    )

    parts.append("EXPERIENCE")
    if title:
        parts.append(f"* {title}")
    previous_titles = _clean_field(row.get("Previous_Job_Titles"))
    parts.extend(f"* {t.strip()}" for t in previous_titles.split(",") if t.strip())

    parts.append("EDUCATION")
    parts.append(_build_education_line(row))

    skills = _clean_field(row.get("Skills"))
    if skills:
        parts.append(f"SKILLS\n{skills}")

    certifications = _clean_field(row.get("Certifications"))
    if certifications:
        parts.append(f"CERTIFICATIONS\n{certifications}")

    return "\n".join(parts)


def _build_education_line(row: pd.Series) -> str:
    """
    Assemble the education line from whichever structured fields are present.

    The graduation year is parsed defensively: a missing year arrives as NaN,
    which is truthy, so a bare truth test followed by int() raises ValueError.
    """
    line = _clean_field(row.get("Degrees")) or _clean_field(row.get("Education_Level"))

    field_of_study = _clean_field(row.get("Field_of_Study"))
    if field_of_study:
        line += f" in {field_of_study}"

    institute = _clean_field(row.get("Institute_Name"))
    if institute:
        line += f" - {institute}"

    graduation_year = _clean_field(row.get("Graduation_Year"))
    if graduation_year:
        try:
            line += f" ({int(float(graduation_year))})"
        except ValueError:
            logger.debug(f"Unparseable graduation year: {graduation_year!r}")

    return line.strip()


# ── Rule-based scoring ────────────────────────────────────────────────────────


def parse_skills(value: Any) -> set[str]:
    """Parse a comma-separated skills field into a lowercase set."""
    text = _clean_field(value)
    if not text:
        return set()
    return {skill.strip().lower() for skill in text.split(",") if skill.strip()}


def score_skill_overlap(resume_skills: set[str], job_text: str) -> float:
    """Fraction of the candidate's skills that the posting mentions."""
    if not resume_skills:
        return _NEUTRAL_SKILLS
    matches = sum(1 for skill in resume_skills if skill in job_text)
    return min(1.0, matches / len(resume_skills))


def score_experience_fit(resume_years: float, required_years: float | None) -> float:
    """
    Score years of experience against the posting's stated requirement.

    Meeting it scores 1.0, being far over is mildly penalised as a seniority
    mismatch, and falling short costs 0.2 per missing year.
    """
    if required_years is None:
        return _NEUTRAL_EXPERIENCE

    gap = resume_years - required_years
    if gap >= 0:
        return 1.0 if gap <= 3 else 0.8
    return max(0.0, 1.0 + gap * 0.2)


def score_education_fit(education_level: str, job_text: str) -> float:
    """Score the candidate's degree against the level the posting asks for."""
    resume_rank = DEGREE_RANK.get(
        detect_education_level(education_level), DEGREE_RANK["none"]
    )
    required_rank = DEGREE_RANK[detect_education_level(job_text, default="bachelors")]

    gap = resume_rank - required_rank
    if gap >= 0:
        return 1.0
    return 0.6 if gap == -1 else 0.3


def score_title_keyword_match(row: pd.Series, job_text: str) -> float:
    """Score how well the candidate's title and field of study appear in the posting."""
    score = 0.2  # Baseline: every pair shares some generic vocabulary.

    title = _clean_field(row.get("Current_Job_Title")).lower()
    if title and title in job_text:
        score += 0.5

    field_of_study = _clean_field(row.get("Field_of_Study")).lower()
    if field_of_study:
        significant = {word for word in field_of_study.split() if len(word) > 3}
        matches = sum(1 for word in significant if word in job_text)
        score += min(0.3, matches * 0.15)

    return min(1.0, score)


def compute_match_score(
    row: pd.Series, job_text: str
) -> tuple[float, dict[str, float]]:
    """
    Score one resume/posting pair across the four components.

    Args:
        row: Structured resume row.
        job_text: Job description text.

    Returns:
        (composite score in [0, 1], per-component breakdown)
    """
    job_lower = job_text.lower()
    try:
        resume_years = float(_clean_field(row.get("Experience_Years")) or 0)
    except ValueError:
        resume_years = 0.0

    breakdown = {
        "skills_match": score_skill_overlap(parse_skills(row.get("Skills")), job_lower),
        "experience_relevance": score_experience_fit(
            resume_years, required_experience_years(job_lower)
        ),
        "education_fit": score_education_fit(
            _clean_field(row.get("Education_Level")), job_lower
        ),
        "keyword_alignment": score_title_keyword_match(row, job_lower),
    }

    composite = sum(_WEIGHTS[name] * value for name, value in breakdown.items())
    return (
        round(min(1.0, max(0.0, composite)), 4),
        {name: round(value, 4) for name, value in breakdown.items()},
    )


# ── Construction ──────────────────────────────────────────────────────────────


def build_eval_dataset(
    source_path: Path,
    jobs_path: Path,
    pairs_per_resume: int,
    jobs_pool_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Pair each structured resume with sampled postings and score every pair.

    Postings are sampled into a fixed pool once, so scoring cost stays linear in
    the number of resumes rather than scanning the full postings table per row.

    Args:
        source_path: Structured resume CSV.
        jobs_path: Cleaned job postings CSV.
        pairs_per_resume: Postings paired with each resume.
        jobs_pool_size: Size of the sampled posting pool.
        random_seed: Seed for reproducible sampling.

    Returns:
        DataFrame of scored pairs.
    """
    rng = np.random.default_rng(random_seed)

    logger.info(f"Loading structured resumes from {source_path}")
    resumes = pd.read_csv(source_path)
    logger.info(f"Loaded {len(resumes)} structured resumes")

    logger.info(f"Loading job postings from {jobs_path}")
    jobs = pd.read_csv(jobs_path, usecols=["description"]).dropna()
    jobs = jobs[jobs["description"].str.len() > MIN_JD_CHARS].reset_index(drop=True)
    if len(jobs) > jobs_pool_size:
        jobs = jobs.sample(n=jobs_pool_size, random_state=random_seed).reset_index(
            drop=True
        )
    logger.info(f"Sampled a pool of {len(jobs)} job postings")

    descriptions = jobs["description"].astype(str).tolist()

    records: list[dict[str, Any]] = []
    for _, row in tqdm(
        resumes.iterrows(), total=len(resumes), desc="Building eval pairs"
    ):
        resume_text = build_resume_text(row)
        category = _clean_field(row.get("Current_Job_Title")) or "Unknown"

        for job_index in rng.integers(len(descriptions), size=pairs_per_resume):
            description = descriptions[int(job_index)]
            score, breakdown = compute_match_score(row, description)
            records.append(
                {
                    "resume_text": resume_text,
                    "job_description": description,
                    "match_score": score,
                    "category": category,
                    "score_breakdown": json.dumps(breakdown),
                }
            )

    pairs = pd.DataFrame(records)
    logger.info(f"Built {len(pairs)} evaluation pairs")
    logger.info(
        f"Score distribution:\n{pairs['match_score'].describe().round(3).to_string()}"
    )
    return pairs


def run_eval_dataset_build() -> pd.DataFrame:
    """
    Build and persist the held-out evaluation set.

    Returns:
        The saved pairs DataFrame.

    Raises:
        FileNotFoundError: If either source dataset is missing.
    """
    logger.info("=" * 60)
    logger.info("Building the held-out evaluation set")
    logger.info("=" * 60)

    if not EVAL_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Structured resume dataset not found at {EVAL_SOURCE_PATH}. Download it with:\n"
            f"  kaggle datasets download {get_section('data')['eval_dataset']} "
            "--path data/raw/eval_dataset --unzip"
        )
    if not JOBS_PATH.exists():
        raise FileNotFoundError(
            f"Job postings CSV not found at {JOBS_PATH}. Run the ingest stage first."
        )

    data_config = get_section("data")
    settings = data_config["eval_set"]

    pairs = build_eval_dataset(
        source_path=EVAL_SOURCE_PATH,
        jobs_path=JOBS_PATH,
        pairs_per_resume=settings["pairs_per_resume"],
        jobs_pool_size=settings["jobs_pool_size"],
        random_seed=data_config["random_seed"],
    )

    before = len(pairs)
    pairs = pairs[
        (pairs["resume_text"].str.len() > 50)
        & (pairs["job_description"].str.len() > 80)
        & pairs["match_score"].between(0.0, 1.0)
    ].reset_index(drop=True)
    logger.info(f"Quality filter: {before} → {len(pairs)} pairs")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(EVAL_PAIRS_PATH, index=False)
    logger.success(f"Wrote {len(pairs)} pairs to {EVAL_PAIRS_PATH}")

    _log_to_mlflow(pairs)
    return pairs


def _log_to_mlflow(pairs: pd.DataFrame) -> None:
    """Record evaluation-set statistics to MLflow. Failures are non-fatal."""
    try:
        mlflow.set_tracking_uri(_mlflow_tracking_uri())
        mlflow.set_experiment(_mlflow_experiment())
        with mlflow.start_run(run_name="eval-dataset-build"):
            mlflow.log_metrics(
                {
                    "eval_pairs_total": float(len(pairs)),
                    "eval_score_mean": float(pairs["match_score"].mean()),
                    "eval_score_std": float(pairs["match_score"].std()),
                    "eval_unique_categories": float(pairs["category"].nunique()),
                }
            )
            mlflow.log_artifact(str(EVAL_PAIRS_PATH), artifact_path="eval")
            mlflow.set_tag("stage", "eval_dataset")
            mlflow.set_tag("source", get_section("data")["eval_dataset"])
            logger.success("Evaluation set statistics logged to MLflow.")
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    configure_logging(log_file="eval_dataset.log")
    run_eval_dataset_build()
