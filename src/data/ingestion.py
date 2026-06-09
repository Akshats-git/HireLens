"""
Data ingestion module for HireLens.

Downloads and registers:
  - Kaggle Resume Dataset (snehaanbhawal/resume-dataset)
  - LinkedIn Job Postings dataset (arshkon/linkedin-job-postings)

All dataset statistics are logged to MLflow.
Raw files are saved to data/raw/ and registered with DVC.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger

# ── Project root resolution ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "hirelens-resume-matching")


# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    level="INFO",
)
logger.add(
    PROJECT_ROOT / "logs" / "ingestion.log",
    rotation="50 MB",
    retention="14 days",
    level="DEBUG",
)


# ── Kaggle helpers ────────────────────────────────────────────────────────────

def _check_kaggle_credentials() -> None:
    """Raise EnvironmentError if Kaggle API credentials are not configured.

    Supports both credential formats:
      - Legacy: ~/.kaggle/kaggle.json  ({"username":..., "key":...})
      - New token: ~/.kaggle/access_token  (KGAT_... string)
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    env_key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_TOKEN")
    env_user = os.getenv("KAGGLE_USERNAME")

    has_json = kaggle_json.exists()
    has_token = access_token.exists() and access_token.stat().st_size > 0
    has_env = bool(env_key) and (bool(env_user) or env_key.startswith("KGAT_"))

    if not (has_json or has_token or has_env):
        raise EnvironmentError(
            "Kaggle credentials not found. Either:\n"
            "  1. New token:  mkdir -p ~/.kaggle && echo KGAT_... > ~/.kaggle/access_token\n"
            "  2. Legacy key: place kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. Env vars:   set KAGGLE_USERNAME and KAGGLE_KEY\n"
            "Get credentials at https://www.kaggle.com/account"
        )
    logger.info("Kaggle credentials verified.")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a shell command and raise on non-zero exit."""
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR: {result.stderr}")
    return result


def download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> Path:
    """
    Download a Kaggle dataset to dest_dir using the Kaggle CLI.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g. 'snehaanbhawal/resume-dataset').
        dest_dir: Local directory to download into.

    Returns:
        Path to the downloaded directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Kaggle dataset: {dataset_slug} → {dest_dir}")

    _run([
        "kaggle", "datasets", "download",
        "--dataset", dataset_slug,
        "--path", str(dest_dir),
        "--unzip",
    ])

    logger.success(f"Dataset {dataset_slug} downloaded to {dest_dir}")
    return dest_dir


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_resume_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Load the Kaggle Resume Dataset (snehaanbhawal/resume-dataset).

    Expected file: data/raw/resume_dataset/Resume.csv
    Columns: ID, Resume_str, Resume_html, Category

    Args:
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with columns [id, resume_text, category].
    """
    resume_dir = raw_dir / "resume_dataset"
    csv_path = resume_dir / "Resume" / "Resume.csv"

    if not csv_path.exists():
        logger.info("Resume dataset not found locally — downloading from Kaggle.")
        _check_kaggle_credentials()
        download_kaggle_dataset("snehaanbhawal/resume-dataset", resume_dir)

    logger.info(f"Loading resume dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalise column names
    df = df.rename(columns={
        "ID": "id",
        "Resume_str": "resume_text",
        "Resume_html": "resume_html",
        "Category": "category",
    })

    # Drop rows with missing text
    before = len(df)
    df = df.dropna(subset=["resume_text", "category"])
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df = df[df["resume_text"].str.len() > 50]
    logger.info(f"Resume dataset: {before} rows → {len(df)} after cleaning.")

    return df[["id", "resume_text", "category"]]


def load_linkedin_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Load the LinkedIn Job Postings dataset (arshkon/linkedin-job-postings).

    Expected file: data/raw/linkedin_jobs/job_descriptions.csv
    Uses columns: job_id, title, description, skills_desc, formatted_experience_level

    Args:
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with columns [job_id, title, description, experience_level].
    """
    jobs_dir = raw_dir / "linkedin_jobs"
    csv_path = jobs_dir / "postings.csv"

    if not csv_path.exists():
        logger.info("LinkedIn dataset not found locally — downloading from Kaggle.")
        _check_kaggle_credentials()
        download_kaggle_dataset("arshkon/linkedin-job-postings", jobs_dir)

    logger.info(f"Loading LinkedIn jobs dataset from {csv_path}")

    # Large file — load in chunks to avoid OOM
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=10_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Keep only needed columns (subset may vary by dataset version)
    keep_cols = ["job_id", "title", "description", "formatted_experience_level"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    if "description" not in df.columns:
        raise ValueError("LinkedIn dataset is missing 'description' column. Check dataset version.")

    before = len(df)
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() > 50]
    logger.info(f"LinkedIn dataset: {before} rows → {len(df)} after cleaning.")

    return df


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_resume_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for the resume dataset."""
    stats: dict[str, Any] = {
        "resume_total_rows": len(df),
        "resume_unique_categories": int(df["category"].nunique()),
        "resume_avg_text_length": float(df["resume_text"].str.len().mean()),
        "resume_min_text_length": int(df["resume_text"].str.len().min()),
        "resume_max_text_length": int(df["resume_text"].str.len().max()),
    }
    category_dist = df["category"].value_counts().to_dict()
    stats["resume_category_distribution"] = category_dist
    return stats


def compute_jobs_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for the LinkedIn jobs dataset."""
    stats: dict[str, Any] = {
        "jobs_total_rows": len(df),
        "jobs_avg_description_length": float(df["description"].str.len().mean()),
        "jobs_min_description_length": int(df["description"].str.len().min()),
        "jobs_max_description_length": int(df["description"].str.len().max()),
    }
    if "formatted_experience_level" in df.columns:
        stats["jobs_experience_level_distribution"] = (
            df["formatted_experience_level"].value_counts().to_dict()
        )
    return stats


def log_stats_to_mlflow(resume_stats: dict, jobs_stats: dict) -> None:
    """Log dataset statistics as MLflow metrics and params."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="data-ingestion"):
        # Scalar metrics
        scalar_keys = [
            "resume_total_rows", "resume_unique_categories",
            "resume_avg_text_length", "resume_min_text_length", "resume_max_text_length",
            "jobs_total_rows", "jobs_avg_description_length",
            "jobs_min_description_length", "jobs_max_description_length",
        ]
        for key in scalar_keys:
            value = resume_stats.get(key) or jobs_stats.get(key)
            if value is not None:
                mlflow.log_metric(key, float(value))

        # Distribution as JSON artifact
        distributions = {
            "resume_category_distribution": resume_stats.get("resume_category_distribution", {}),
            "jobs_experience_level_distribution": jobs_stats.get("jobs_experience_level_distribution", {}),
        }
        dist_path = PROJECT_ROOT / "logs" / "dataset_distributions.json"
        dist_path.parent.mkdir(exist_ok=True)
        dist_path.write_text(json.dumps(distributions, indent=2))
        mlflow.log_artifact(str(dist_path), artifact_path="ingestion")

        mlflow.set_tag("stage", "ingestion")
        logger.success("Dataset statistics logged to MLflow.")


# ── DVC registration ──────────────────────────────────────────────────────────

def register_with_dvc(paths: list[Path]) -> None:
    """
    Track raw data files/directories with DVC.

    Args:
        paths: List of paths to add to DVC tracking.
    """
    for path in paths:
        if not path.exists():
            logger.warning(f"DVC: skipping {path} (does not exist).")
            continue
        try:
            _run(["dvc", "add", str(path.relative_to(PROJECT_ROOT))], cwd=PROJECT_ROOT)
            logger.info(f"DVC: tracking {path}")
        except RuntimeError as e:
            logger.warning(f"DVC add failed for {path}: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ingestion pipeline: download → load → stats → DVC → MLflow.

    Returns:
        Tuple of (resume_df, jobs_df).
    """
    logger.info("=" * 60)
    logger.info("Starting HireLens data ingestion pipeline")
    logger.info("=" * 60)

    # Load datasets
    resume_df = load_resume_dataset(RAW_DIR)
    jobs_df = load_linkedin_dataset(RAW_DIR)

    # Compute and display stats
    resume_stats = compute_resume_stats(resume_df)
    jobs_stats = compute_jobs_stats(jobs_df)

    logger.info("Resume dataset stats:")
    for k, v in resume_stats.items():
        if k != "resume_category_distribution":
            logger.info(f"  {k}: {v}")

    logger.info("LinkedIn jobs stats:")
    for k, v in jobs_stats.items():
        if k != "jobs_experience_level_distribution":
            logger.info(f"  {k}: {v}")

    # Save cleaned raw CSVs
    resume_out = RAW_DIR / "resumes_clean.csv"
    jobs_out = RAW_DIR / "jobs_clean.csv"
    resume_df.to_csv(resume_out, index=False)
    jobs_df.to_csv(jobs_out, index=False)
    logger.info(f"Saved: {resume_out}, {jobs_out}")

    # Register with DVC
    register_with_dvc([RAW_DIR / "resume_dataset", RAW_DIR / "linkedin_jobs"])

    # Log to MLflow
    try:
        log_stats_to_mlflow(resume_stats, jobs_stats)
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.success("Data ingestion complete.")
    return resume_df, jobs_df


if __name__ == "__main__":
    run_ingestion()
