"""
Evaluation dataset builder for HireLens.

Uses the Kaggle structured resume dataset (sayyedfaizan95/resume-and-job-description)
paired with real LinkedIn job descriptions to build a held-out evaluation set.

Each row in the source dataset contains structured resume fields (skills, education,
experience, job title). We:
  1. Convert structured fields → formatted resume text
  2. Pair each resume with a LinkedIn job description matched by skills/title
  3. Compute a rule-based match score across 4 dimensions:
       - Skill overlap        (40%)
       - Experience fit       (30%)
       - Education fit        (15%)
       - Title/keyword match  (15%)
  4. Also generate hard-negative pairs (same resume, wrong-domain job) at score ~0.0–0.3

Output: data/synthetic/synthetic_pairs.csv
Columns: resume_text, job_description, match_score, category, score_breakdown
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "hirelens-resume-matching")

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}")
logger.add(PROJECT_ROOT / "logs" / "synthetic_gen.log", rotation="50 MB", retention="14 days", level="DEBUG")

# ── Education degree hierarchy ────────────────────────────────────────────────
_DEGREE_RANK = {
    "high school": 1, "diploma": 2, "certificate": 3,
    "associate": 3, "bachelor": 4, "bachelor's": 4,
    "master": 5, "master's": 5, "mba": 5, "phd": 6, "doctorate": 6,
}

_DEGREE_KEYWORDS = {
    "phd": ["phd", "doctorate", "doctoral", "research"],
    "master's": ["master", "mba", "ms ", "m.s", "graduate degree"],
    "bachelor's": ["bachelor", "bs ", "b.s", "undergraduate", "degree"],
    "diploma": ["diploma", "associate", "certificate"],
}


# ── Resume text builder ───────────────────────────────────────────────────────

def build_resume_text(row: pd.Series) -> str:
    """
    Convert a structured resume row into a natural-language resume text.

    Args:
        row: A row from the structured Kaggle dataset.

    Returns:
        Formatted resume string suitable for embedding.
    """
    parts: list[str] = []

    # Header
    name = str(row.get("Name", "Candidate")).strip()
    parts.append(f"{name}\n")

    # Summary / objective
    current_title = str(row.get("Current_Job_Title", "")).strip()
    exp_years = row.get("Experience_Years", 0)
    if current_title and current_title.lower() != "none":
        parts.append(
            f"PROFESSIONAL SUMMARY\n"
            f"Experienced {current_title} with {exp_years} year(s) of professional experience."
        )
    else:
        parts.append(
            f"PROFESSIONAL SUMMARY\n"
            f"Recent graduate with {exp_years} year(s) of experience seeking new opportunities."
        )

    # Experience
    prev_titles = str(row.get("Previous_Job_Titles", "")).strip()
    parts.append("EXPERIENCE")
    if current_title and current_title.lower() != "none":
        parts.append(f"• {current_title}")
    if prev_titles and prev_titles.lower() not in ("none", "nan", ""):
        for title in prev_titles.split(","):
            parts.append(f"• {title.strip()}")

    # Education
    edu_level = str(row.get("Education_Level", "")).strip()
    field = str(row.get("Field_of_Study", "")).strip()
    degree = str(row.get("Degrees", "")).strip()
    institute = str(row.get("Institute_Name", "")).strip()
    grad_year = row.get("Graduation_Year", "")
    parts.append("EDUCATION")
    edu_line = f"{degree}" if degree and degree.lower() != "nan" else edu_level
    if field and field.lower() != "nan":
        edu_line += f" in {field}"
    if institute and institute.lower() != "nan":
        edu_line += f" — {institute}"
    if grad_year:
        edu_line += f" ({int(grad_year)})"
    parts.append(edu_line)

    # Skills
    skills_raw = str(row.get("Skills", "")).strip()
    if skills_raw and skills_raw.lower() != "nan":
        parts.append(f"SKILLS\n{skills_raw}")

    # Certifications
    certs = str(row.get("Certifications", "")).strip()
    if certs and certs.lower() not in ("none", "nan", ""):
        parts.append(f"CERTIFICATIONS\n{certs}")

    return "\n".join(parts)


# ── Scoring functions ─────────────────────────────────────────────────────────

def _parse_skills(skills_str: str) -> set[str]:
    """Parse comma-separated skills string into a lowercase set."""
    if not skills_str or str(skills_str).lower() in ("nan", "none", ""):
        return set()
    return {s.strip().lower() for s in str(skills_str).split(",") if s.strip()}


def score_skill_overlap(resume_skills: set[str], job_text: str) -> float:
    """
    Compute skill overlap score: fraction of resume skills found in job description.

    Args:
        resume_skills: Set of lowercased skill tokens from the resume.
        job_text: Lowercased job description text.

    Returns:
        Float in [0, 1].
    """
    if not resume_skills:
        return 0.3  # neutral when unknown
    matches = sum(1 for skill in resume_skills if skill in job_text)
    return min(1.0, matches / max(len(resume_skills), 1))


def score_experience_fit(resume_years: float, job_text: str) -> float:
    """
    Score experience fit based on years mentioned in the job description.

    Args:
        resume_years: Years of experience from the resume row.
        job_text: Lowercased job description text.

    Returns:
        Float in [0, 1].
    """
    # Extract required years from JD (e.g. "3+ years", "5 years of experience")
    matches = re.findall(r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?", job_text)
    if not matches:
        return 0.6  # neutral when not specified

    required = float(matches[0])
    gap = resume_years - required

    if gap >= 0:
        return 1.0 if gap <= 3 else 0.8   # overqualified slightly penalised
    else:
        return max(0.0, 1.0 + gap * 0.2)  # underqualified: -0.2 per missing year


def score_education_fit(edu_level: str, job_text: str) -> float:
    """
    Score education fit by matching degree level to job description requirements.

    Args:
        edu_level: Education level string from the resume.
        job_text: Lowercased job description text.

    Returns:
        Float in [0, 1].
    """
    resume_rank = _DEGREE_RANK.get(edu_level.lower(), 3)

    # Determine what the JD requires
    required_rank = 3  # default: bachelor level
    for degree, keywords in _DEGREE_KEYWORDS.items():
        if any(kw in job_text for kw in keywords):
            required_rank = _DEGREE_RANK.get(degree, 3)
            break

    gap = resume_rank - required_rank
    if gap >= 0:
        return 1.0
    elif gap == -1:
        return 0.6
    else:
        return 0.3


def score_title_keyword_match(resume_row: pd.Series, job_text: str) -> float:
    """
    Score title/keyword alignment between resume job title and JD text.

    Args:
        resume_row: Resume row with Current_Job_Title and Field_of_Study.
        job_text: Lowercased job description text.

    Returns:
        Float in [0, 1].
    """
    title = str(resume_row.get("Current_Job_Title", "")).lower()
    field = str(resume_row.get("Field_of_Study", "")).lower()

    score = 0.2  # baseline
    if title and title != "none" and title in job_text:
        score += 0.5
    if field and field != "nan":
        # Check word-level overlap
        field_words = {w for w in field.split() if len(w) > 3}
        matches = sum(1 for w in field_words if w in job_text)
        score += min(0.3, matches * 0.15)

    return min(1.0, score)


def compute_match_score(
    resume_row: pd.Series,
    job_text: str,
) -> tuple[float, dict[str, float]]:
    """
    Compute the composite match score for a resume-JD pair.

    Weights (from configs/config.yaml):
        skills_match:          0.40
        experience_relevance:  0.30
        education_fit:         0.15
        keyword_alignment:     0.15

    Args:
        resume_row: Row from the structured resume dataset.
        job_text: Job description text (will be lowercased internally).

    Returns:
        Tuple of (composite_score, breakdown_dict).
    """
    job_lower = job_text.lower()
    resume_skills = _parse_skills(str(resume_row.get("Skills", "")))
    exp_years = float(resume_row.get("Experience_Years", 0) or 0)
    edu_level = str(resume_row.get("Education_Level", "Bachelor's"))

    s_skills = score_skill_overlap(resume_skills, job_lower)
    s_exp = score_experience_fit(exp_years, job_lower)
    s_edu = score_education_fit(edu_level, job_lower)
    s_kw = score_title_keyword_match(resume_row, job_lower)

    composite = (
        0.40 * s_skills +
        0.30 * s_exp +
        0.15 * s_edu +
        0.15 * s_kw
    )
    composite = round(min(1.0, max(0.0, composite)), 4)

    breakdown = {
        "skills_match": round(s_skills, 4),
        "experience_relevance": round(s_exp, 4),
        "education_fit": round(s_edu, 4),
        "keyword_alignment": round(s_kw, 4),
    }
    return composite, breakdown


# ── Main builder ──────────────────────────────────────────────────────────────

def build_eval_dataset(
    eval_source_path: Path,
    jobs_path: Path,
    neg_ratio: int = 1,
    random_seed: int = 42,
    jobs_sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Build the held-out evaluation dataset from the structured Kaggle resume data
    paired with real LinkedIn job descriptions.

    Samples a fixed pool of jobs upfront (jobs_sample_size) to avoid
    per-resume full-scan over 123K rows.
    """
    rng = np.random.default_rng(random_seed)

    logger.info(f"Loading structured resumes from {eval_source_path}")
    source_df = pd.read_csv(eval_source_path)
    logger.info(f"Loaded {len(source_df)} structured resumes")

    logger.info(f"Loading job descriptions from {jobs_path}")
    jobs_df = pd.read_csv(jobs_path, usecols=["description"]).dropna()
    jobs_df = jobs_df[jobs_df["description"].str.len() > 100].reset_index(drop=True)

    # Sample a manageable pool upfront — avoids O(n*m) regex scans
    if len(jobs_df) > jobs_sample_size:
        jobs_df = jobs_df.sample(n=jobs_sample_size, random_state=random_seed).reset_index(drop=True)
    logger.info(f"Using {len(jobs_df)} job descriptions (sampled pool)")

    records: list[dict[str, Any]] = []

    for _, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Building eval pairs"):
        resume_text = build_resume_text(row)
        category = str(row.get("Current_Job_Title", "Unknown")).strip()

        # ── Positive pair: random sample, scored by rule-based scorer ─────────
        pos_idx = int(rng.integers(len(jobs_df)))
        jd = str(jobs_df.iloc[pos_idx]["description"])
        score, breakdown = compute_match_score(row, jd)
        records.append({
            "resume_text": resume_text,
            "job_description": jd,
            "match_score": score,
            "category": category,
            "pair_type": "scored",
            "score_breakdown": json.dumps(breakdown),
        })

        # ── Additional pairs for neg_ratio > 1 ───────────────────────────────
        for _ in range(neg_ratio):
            neg_idx = int(rng.integers(len(jobs_df)))
            jd = str(jobs_df.iloc[neg_idx]["description"])
            score, breakdown = compute_match_score(row, jd)
            records.append({
                "resume_text": resume_text,
                "job_description": jd,
                "match_score": score,
                "category": category,
                "pair_type": "scored",
                "score_breakdown": json.dumps(breakdown),
            })

    df = pd.DataFrame(records)
    logger.info(f"Built {len(df)} eval pairs")
    logger.info(f"Score distribution:\n{df['match_score'].describe().round(3).to_string()}")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_synthetic_generation(n_pairs: int = 500) -> pd.DataFrame:
    """
    Build the evaluation dataset from the Kaggle structured resume dataset.

    Uses rule-based scoring across 4 dimensions (skills, experience, education,
    keywords) instead of LLM generation — free, fast, and interpretable.

    Args:
        n_pairs: Approximate target pairs (dataset has 1200 resumes × neg_ratio+1).

    Returns:
        Saved evaluation pairs DataFrame.
    """
    logger.info("=" * 60)
    logger.info("Building evaluation dataset from Kaggle structured resumes")
    logger.info("=" * 60)

    eval_source = PROJECT_ROOT / "data" / "raw" / "eval_dataset" / "resume_dataset_1200.csv"
    jobs_path = PROJECT_ROOT / "data" / "raw" / "jobs_clean.csv"

    if not eval_source.exists():
        raise FileNotFoundError(
            f"Eval source not found at {eval_source}. "
            "Download it with: kaggle datasets download sayyedfaizan95/resume-and-job-description "
            "--path data/raw/eval_dataset --unzip"
        )
    if not jobs_path.exists():
        raise FileNotFoundError(
            f"Jobs CSV not found at {jobs_path}. Run ingestion first."
        )

    df = build_eval_dataset(
        eval_source_path=eval_source,
        jobs_path=jobs_path,
        neg_ratio=1,
    )

    # Quality filter
    before = len(df)
    df = df[
        (df["resume_text"].str.len() > 50) &
        (df["job_description"].str.len() > 80) &
        (df["match_score"].between(0.0, 1.0))
    ].reset_index(drop=True)
    logger.info(f"Quality filter: {before} → {len(df)} pairs")

    output_path = SYNTHETIC_DIR / "synthetic_pairs.csv"
    df.to_csv(output_path, index=False)
    logger.success(f"Saved {len(df)} eval pairs to {output_path}")

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="eval-dataset-generation"):
            mlflow.log_metrics({
                "eval_pairs_total": float(len(df)),
                "eval_positive_pairs": float((df["pair_type"] == "positive").sum()),
                "eval_negative_pairs": float((df["pair_type"] == "negative").sum()),
                "eval_score_mean": float(df["match_score"].mean()),
                "eval_score_std": float(df["match_score"].std()),
                "eval_unique_categories": float(df["category"].nunique()),
            })
            mlflow.log_artifact(str(output_path), artifact_path="eval")
            mlflow.set_tag("stage", "eval_dataset_generation")
            mlflow.set_tag("source", "kaggle:sayyedfaizan95/resume-and-job-description")
            logger.success("Eval dataset stats logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.success(f"Evaluation dataset complete: {len(df)} pairs.")
    return df


if __name__ == "__main__":
    run_synthetic_generation()
