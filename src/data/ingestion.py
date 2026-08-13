"""
Dataset ingestion.

Downloads and normalises the two source datasets named in configs/config.yaml:
  - a Kaggle resume dataset, categorised by profession
  - the LinkedIn job postings dataset

Cleaned CSVs are written to data/raw/, the raw downloads are registered with DVC,
and summary statistics are logged to MLflow.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, get_section
from src.logging_setup import configure_logging

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DISTRIBUTIONS_PATH = PROJECT_ROOT / "logs" / "dataset_distributions.json"

RESUMES_CSV = RAW_DIR / "resumes_clean.csv"
JOBS_CSV = RAW_DIR / "jobs_clean.csv"

# Rows shorter than this are boilerplate or extraction failures.
MIN_TEXT_CHARS = 50

# The postings file is large enough that a single read can exhaust memory.
READ_CHUNK_ROWS = 10_000


def _mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _mlflow_experiment() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", get_section("mlflow")["experiment_name"])


# ── Kaggle ────────────────────────────────────────────────────────────────────


def check_kaggle_credentials() -> None:
    """
    Verify Kaggle API credentials are configured.

    Accepts the legacy ~/.kaggle/kaggle.json, the newer ~/.kaggle/access_token,
    or environment variables.

    Raises:
        EnvironmentError: If no usable credential source is present.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    env_key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_TOKEN")

    has_json = kaggle_json.exists()
    has_token = access_token.exists() and access_token.stat().st_size > 0
    has_env = bool(env_key) and (
        bool(os.getenv("KAGGLE_USERNAME")) or env_key.startswith("KGAT_")
    )

    if not (has_json or has_token or has_env):
        raise EnvironmentError(
            "Kaggle credentials not found. Use any one of:\n"
            "  1. New token:  mkdir -p ~/.kaggle && echo KGAT_... > ~/.kaggle/access_token\n"
            "  2. Legacy key: place kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. Env vars:   set KAGGLE_USERNAME and KAGGLE_KEY\n"
            "Credentials are available at https://www.kaggle.com/account"
        )
    logger.info("Kaggle credentials verified.")


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, raising RuntimeError with its stderr on failure."""
    logger.debug(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDERR: {result.stderr}"
        )
    return result


def download_kaggle_dataset(dataset_slug: str, destination: Path) -> Path:
    """
    Download and unzip a Kaggle dataset via the Kaggle CLI.

    Args:
        dataset_slug: Dataset identifier, e.g. 'arshkon/linkedin-job-postings'.
        destination: Directory to download into.

    Returns:
        The destination directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Kaggle dataset {dataset_slug} → {destination}")

    _run(
        [
            "kaggle",
            "datasets",
            "download",
            "--dataset",
            dataset_slug,
            "--path",
            str(destination),
            "--unzip",
        ]
    )

    logger.success(f"Downloaded {dataset_slug}")
    return destination


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_resume_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Load the categorised resume dataset, downloading it if absent.

    Args:
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with columns [id, resume_text, category].
    """
    dataset_dir = raw_dir / "resume_dataset"
    csv_path = dataset_dir / "Resume" / "Resume.csv"

    if not csv_path.exists():
        logger.info("Resume dataset not found locally — downloading from Kaggle.")
        check_kaggle_credentials()
        download_kaggle_dataset(get_section("data")["resume_dataset"], dataset_dir)

    logger.info(f"Loading resume dataset from {csv_path}")
    df = pd.read_csv(csv_path).rename(
        columns={
            "ID": "id",
            "Resume_str": "resume_text",
            "Resume_html": "resume_html",
            "Category": "category",
        }
    )

    before = len(df)
    df = df.dropna(subset=["resume_text", "category"])
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df = df[df["resume_text"].str.len() > MIN_TEXT_CHARS]
    logger.info(f"Resume dataset: {before} rows → {len(df)} after cleaning.")

    return df[["id", "resume_text", "category"]]


def load_jobs_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Load the LinkedIn job postings dataset, downloading it if absent.

    Args:
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with the available subset of
        [job_id, title, description, formatted_experience_level].

    Raises:
        ValueError: If the file has no 'description' column.
    """
    dataset_dir = raw_dir / "linkedin_jobs"
    csv_path = dataset_dir / "postings.csv"

    if not csv_path.exists():
        logger.info("LinkedIn dataset not found locally — downloading from Kaggle.")
        check_kaggle_credentials()
        download_kaggle_dataset(get_section("data")["jobs_dataset"], dataset_dir)

    logger.info(f"Loading job postings from {csv_path}")
    chunks = pd.read_csv(csv_path, chunksize=READ_CHUNK_ROWS, low_memory=False)
    df = pd.concat(chunks, ignore_index=True)

    if "description" not in df.columns:
        raise ValueError(
            "Job postings dataset has no 'description' column — check the dataset version."
        )

    wanted = ["job_id", "title", "description", "formatted_experience_level"]
    df = df[[column for column in wanted if column in df.columns]].copy()

    before = len(df)
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() > MIN_TEXT_CHARS]
    logger.info(f"Job postings: {before} rows → {len(df)} after cleaning.")

    return df


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_resume_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise the resume dataset."""
    lengths = df["resume_text"].str.len()
    return {
        "resume_total_rows": len(df),
        "resume_unique_categories": int(df["category"].nunique()),
        "resume_avg_text_length": float(lengths.mean()),
        "resume_min_text_length": int(lengths.min()),
        "resume_max_text_length": int(lengths.max()),
        "resume_category_distribution": df["category"].value_counts().to_dict(),
    }


def compute_jobs_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise the job postings dataset."""
    lengths = df["description"].str.len()
    stats: dict[str, Any] = {
        "jobs_total_rows": len(df),
        "jobs_avg_description_length": float(lengths.mean()),
        "jobs_min_description_length": int(lengths.min()),
        "jobs_max_description_length": int(lengths.max()),
    }
    if "formatted_experience_level" in df.columns:
        stats["jobs_experience_level_distribution"] = (
            df["formatted_experience_level"].value_counts().to_dict()
        )
    return stats


def log_stats_to_mlflow(resume_stats: dict, jobs_stats: dict) -> None:
    """
    Log dataset statistics as MLflow metrics, with distributions as an artifact.

    The two stat dicts use disjoint key prefixes and are merged rather than
    probed one after the other: a lookup chained with `or` would treat a
    legitimate zero as missing and fall through to the other dataset's value.
    """
    mlflow.set_tracking_uri(_mlflow_tracking_uri())
    mlflow.set_experiment(_mlflow_experiment())

    merged = {**resume_stats, **jobs_stats}
    scalars = {
        key: float(value)
        for key, value in merged.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    with mlflow.start_run(run_name="data-ingestion"):
        mlflow.log_metrics(scalars)

        distributions = {
            "resume_category_distribution": resume_stats.get(
                "resume_category_distribution", {}
            ),
            "jobs_experience_level_distribution": jobs_stats.get(
                "jobs_experience_level_distribution", {}
            ),
        }
        DISTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISTRIBUTIONS_PATH.write_text(json.dumps(distributions, indent=2))
        mlflow.log_artifact(str(DISTRIBUTIONS_PATH), artifact_path="ingestion")

        mlflow.set_tag("stage", "ingestion")
        logger.success("Dataset statistics logged to MLflow.")


# ── DVC ───────────────────────────────────────────────────────────────────────


def register_with_dvc(paths: list[Path]) -> None:
    """Track raw dataset directories with DVC, warning rather than failing."""
    for path in paths:
        if not path.exists():
            logger.warning(f"DVC: skipping {path} (does not exist).")
            continue
        try:
            _run(["dvc", "add", str(path.relative_to(PROJECT_ROOT))])
            logger.info(f"DVC: now tracking {path}")
        except RuntimeError as exc:
            logger.warning(f"DVC add failed for {path}: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download, clean, and register both datasets.

    Returns:
        (resume_df, jobs_df) as written to data/raw/.
    """
    logger.info("=" * 60)
    logger.info("HireLens data ingestion")
    logger.info("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    resume_df = load_resume_dataset(RAW_DIR)
    jobs_df = load_jobs_dataset(RAW_DIR)

    resume_stats = compute_resume_stats(resume_df)
    jobs_stats = compute_jobs_stats(jobs_df)
    for label, stats in (("Resume", resume_stats), ("Job postings", jobs_stats)):
        logger.info(f"{label} statistics:")
        for key, value in stats.items():
            if not key.endswith("_distribution"):
                logger.info(f"  {key}: {value}")

    resume_df.to_csv(RESUMES_CSV, index=False)
    jobs_df.to_csv(JOBS_CSV, index=False)
    logger.info(f"Wrote {RESUMES_CSV} and {JOBS_CSV}")

    register_with_dvc([RAW_DIR / "resume_dataset", RAW_DIR / "linkedin_jobs"])

    try:
        log_stats_to_mlflow(resume_stats, jobs_stats)
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    logger.success("Data ingestion complete.")
    return resume_df, jobs_df


if __name__ == "__main__":
    configure_logging(log_file="ingestion.log")
    run_ingestion()
