"""
Data pipeline orchestrator.

Runs the data stages in order:
  1. ingest    — download and clean the source datasets
  2. preprocess — clean, section, and pair resumes with postings
  3. eval-set   — build the held-out evaluation set

Usage:
  python -m src.data.data_pipeline                    # every stage
  python -m src.data.data_pipeline --stage ingest
  python -m src.data.data_pipeline --stage preprocess
  python -m src.data.data_pipeline --stage eval-set
  python -m src.data.data_pipeline --skip-eval-set
"""

import argparse
import json
import os
import time

import mlflow
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, get_section
from src.logging_setup import configure_logging

STAGES = ("ingest", "preprocess", "eval-set")

DISTRIBUTIONS_PATH = PROJECT_ROOT / "logs" / "dataset_distributions.json"

# Files each stage is expected to produce, for the post-run check.
_EXPECTED_OUTPUTS = {
    "data/raw/resumes_clean.csv": "ingest",
    "data/raw/jobs_clean.csv": "ingest",
    "data/processed/train_pairs.csv": "preprocess",
    "data/processed/val_pairs.csv": "preprocess",
    "data/processed/test_pairs.csv": "preprocess",
    "data/eval/eval_pairs.csv": "eval-set",
}


def _banner(title: str) -> None:
    logger.info("")
    logger.info("━" * 60)
    logger.info(f"  {title}")
    logger.info("━" * 60)


# ── Stages ────────────────────────────────────────────────────────────────────


def stage_ingest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and clean the source datasets, then write the DVC metrics file."""
    _banner("Stage 1/3 — Data ingestion")
    from src.data.ingestion import run_ingestion

    started = time.perf_counter()
    resume_df, jobs_df = run_ingestion()
    elapsed = time.perf_counter() - started

    DISTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISTRIBUTIONS_PATH.write_text(
        json.dumps(
            {
                "resume_total_rows": int(len(resume_df)),
                "resume_unique_categories": int(resume_df["category"].nunique()),
                "resume_avg_text_length": float(
                    resume_df["resume_text"].str.len().mean()
                ),
                "jobs_total_rows": int(len(jobs_df)),
                "jobs_avg_description_length": float(
                    jobs_df["description"].str.len().mean()
                ),
                "ingestion_elapsed_s": round(elapsed, 2),
            },
            indent=2,
        )
    )

    logger.success(
        f"Ingestion complete in {elapsed:.1f}s — "
        f"{len(resume_df)} resumes, {len(jobs_df)} postings."
    )
    return resume_df, jobs_df


def stage_preprocess(
    resume_df: pd.DataFrame | None = None, jobs_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Clean, section, and pair the datasets."""
    _banner("Stage 2/3 — Preprocessing and pair construction")
    from src.data.preprocessing import run_preprocessing

    started = time.perf_counter()
    pairs_df = run_preprocessing(resume_df=resume_df, jobs_df=jobs_df)
    logger.success(
        f"Preprocessing complete in {time.perf_counter() - started:.1f}s — "
        f"{len(pairs_df)} pairs."
    )
    return pairs_df


def stage_eval_set() -> pd.DataFrame:
    """Build the held-out evaluation set."""
    _banner("Stage 3/3 — Evaluation set construction")
    from src.data.eval_dataset import run_eval_dataset_build

    started = time.perf_counter()
    pairs = run_eval_dataset_build()
    logger.success(
        f"Evaluation set complete in {time.perf_counter() - started:.1f}s — "
        f"{len(pairs)} pairs."
    )
    return pairs


# ── Reporting ─────────────────────────────────────────────────────────────────


def _log_summary(stages_run: list[str], elapsed: float) -> None:
    """Record an end-to-end pipeline run in MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", get_section("mlflow")["experiment_name"])
    )
    with mlflow.start_run(run_name="data-pipeline-summary"):
        mlflow.log_metric("pipeline_total_seconds", elapsed)
        mlflow.log_param("stages_run", ",".join(stages_run))
        mlflow.set_tag("stage", "pipeline_summary")


def _verify_outputs(stages_run: list[str]) -> None:
    """Report on the expected outputs of the stages that ran. Never raises."""
    _banner("Output verification")

    missing = False
    for relative_path, stage in _EXPECTED_OUTPUTS.items():
        if stage not in stages_run:
            continue
        path = PROJECT_ROOT / relative_path
        if path.exists():
            size_kb = path.stat().st_size / 1024
            logger.info(f"  ok       {relative_path:<45} ({size_kb:.1f} KB)")
        else:
            logger.warning(f"  MISSING  {relative_path:<45} [{stage}]")
            missing = True

    if missing:
        logger.warning("Some expected outputs are missing — check the log above.")
    else:
        logger.success("All expected outputs are present.")


# ── Entry point ───────────────────────────────────────────────────────────────


def run_pipeline(stages: list[str] | None = None, skip_eval_set: bool = False) -> None:
    """
    Run the requested pipeline stages in order.

    Args:
        stages: Subset of STAGES to run. Runs all of them if omitted.
        skip_eval_set: Skip evaluation set construction.
    """
    if stages is None:
        stages = [s for s in STAGES if not (skip_eval_set and s == "eval-set")]

    logger.info("=" * 60)
    logger.info(f"HireLens data pipeline — stages: {', '.join(stages)}")
    logger.info("=" * 60)

    started = time.perf_counter()
    stages_run: list[str] = []
    resume_df = jobs_df = None

    if "ingest" in stages:
        resume_df, jobs_df = stage_ingest()
        stages_run.append("ingest")

    if "preprocess" in stages:
        stage_preprocess(resume_df=resume_df, jobs_df=jobs_df)
        stages_run.append("preprocess")

    if "eval-set" in stages:
        stage_eval_set()
        stages_run.append("eval-set")

    elapsed = time.perf_counter() - started

    try:
        _log_summary(stages_run, elapsed)
    except Exception as exc:
        logger.warning(f"MLflow summary logging failed (non-fatal): {exc}")

    _verify_outputs(stages_run)
    logger.success(f"Pipeline finished in {elapsed:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HireLens data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", choices=STAGES, help="Run a single stage instead of all of them"
    )
    parser.add_argument(
        "--skip-eval-set",
        action="store_true",
        help="Run ingest and preprocess only",
    )
    args = parser.parse_args()

    configure_logging(log_file="pipeline.log")
    run_pipeline(
        stages=[args.stage] if args.stage else None,
        skip_eval_set=args.skip_eval_set,
    )


if __name__ == "__main__":
    main()
