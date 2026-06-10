"""
HireLens master data pipeline.

Runs the full pipeline in sequence:
  Stage 1 — ingestion.py   : Download datasets, register with DVC
  Stage 2 — preprocessing.py: Clean, detect sections, build training pairs
  Stage 3 — synthetic_gen.py: Generate synthetic evaluation pairs

Usage:
  python -m src.data.data_pipeline               # all stages
  python -m src.data.data_pipeline --stage ingest
  python -m src.data.data_pipeline --stage preprocess
  python -m src.data.data_pipeline --stage synthetic --n-pairs 500
  python -m src.data.data_pipeline --skip-synthetic   # skip API calls
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}")
logger.add(PROJECT_ROOT / "logs" / "pipeline.log", rotation="100 MB", retention="30 days", level="DEBUG")


def _section(title: str) -> None:
    """Print a visible section separator to the log."""
    logger.info("")
    logger.info("━" * 60)
    logger.info(f"  {title}")
    logger.info("━" * 60)


def stage_ingest() -> tuple:
    """Run Stage 1: data ingestion."""
    _section("Stage 1 / 3 — Data Ingestion")
    from src.data.ingestion import run_ingestion
    t0 = time.perf_counter()
    resume_df, jobs_df = run_ingestion()
    elapsed = time.perf_counter() - t0

    # Write DVC-tracked metric file
    dist = {
        "resume_total_rows": int(len(resume_df)),
        "resume_unique_categories": int(resume_df["category"].nunique()) if "category" in resume_df.columns else 0,
        "resume_avg_text_length": float(resume_df["resume_text"].str.len().mean()) if "resume_text" in resume_df.columns else 0.0,
        "jobs_total_rows": int(len(jobs_df)),
        "jobs_avg_description_length": float(jobs_df["description"].str.len().mean()) if "description" in jobs_df.columns else 0.0,
        "ingestion_elapsed_s": round(elapsed, 2),
    }
    dist_path = PROJECT_ROOT / "logs" / "dataset_distributions.json"
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dist_path, "w") as f:
        json.dump(dist, f, indent=2)
    logger.info(f"Written: {dist_path}")

    logger.success(f"Ingestion complete in {elapsed:.1f}s — "
                   f"{len(resume_df)} resumes, {len(jobs_df)} jobs.")
    return resume_df, jobs_df


def stage_preprocess(resume_df=None, jobs_df=None) -> object:
    """Run Stage 2: preprocessing and pair construction."""
    _section("Stage 2 / 3 — Preprocessing & Pair Construction")
    from src.data.preprocessing import run_preprocessing
    t0 = time.perf_counter()
    pairs_df = run_preprocessing(resume_df=resume_df, jobs_df=jobs_df)
    elapsed = time.perf_counter() - t0
    logger.success(f"Preprocessing complete in {elapsed:.1f}s — {len(pairs_df)} training pairs.")
    return pairs_df


def stage_synthetic(n_pairs: int = 500) -> object:
    """Run Stage 3: synthetic evaluation pair generation."""
    _section("Stage 3 / 3 — Synthetic Evaluation Data Generation")
    from src.data.synthetic_gen import run_synthetic_generation
    t0 = time.perf_counter()
    synthetic_df = run_synthetic_generation(n_pairs=n_pairs)
    elapsed = time.perf_counter() - t0
    logger.success(f"Synthetic generation complete in {elapsed:.1f}s — {len(synthetic_df)} pairs.")
    return synthetic_df


def log_pipeline_summary(stages_run: list[str], total_elapsed: float) -> None:
    """Log an end-to-end pipeline summary run to MLflow."""
    import os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "hirelens-resume-matching"))

    with mlflow.start_run(run_name="data-pipeline-summary"):
        mlflow.log_metric("pipeline_total_seconds", total_elapsed)
        mlflow.log_param("stages_run", ",".join(stages_run))
        mlflow.set_tag("stage", "pipeline_summary")
        mlflow.set_tag("status", "success")


def _verify_outputs() -> None:
    """
    Quick post-run sanity check — log whether expected output files exist.
    Logs warnings but does not raise exceptions.
    """
    expected = {
        "data/raw/resumes_clean.csv": "Ingestion",
        "data/raw/jobs_clean.csv": "Ingestion",
        "data/processed/train_pairs.csv": "Preprocessing",
        "data/processed/val_pairs.csv": "Preprocessing",
        "data/processed/test_pairs.csv": "Preprocessing",
    }
    _section("Output Verification")
    all_ok = True
    for rel_path, stage in expected.items():
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            logger.info(f"  ✓  {rel_path:<45} ({size_kb:.1f} KB)  [{stage}]")
        else:
            logger.warning(f"  ✗  {rel_path:<45} MISSING  [{stage}]")
            all_ok = False

    synthetic_path = PROJECT_ROOT / "data/synthetic/synthetic_pairs.csv"
    if synthetic_path.exists():
        size_kb = synthetic_path.stat().st_size / 1024
        logger.info(f"  ✓  {'data/synthetic/synthetic_pairs.csv':<45} ({size_kb:.1f} KB)  [Synthetic]")
    else:
        logger.info(f"  -  {'data/synthetic/synthetic_pairs.csv':<45} (skipped or not run)")

    if all_ok:
        logger.success("All expected outputs present.")
    else:
        logger.warning("Some outputs are missing — check logs above.")


def run_pipeline(
    stages: list[str] | None = None,
    skip_synthetic: bool = False,
    n_synthetic: int = 500,
) -> None:
    """
    Run the full HireLens data pipeline or selected stages.

    Args:
        stages: List of stages to run: ['ingest', 'preprocess', 'synthetic'].
                Runs all three if None.
        skip_synthetic: Skip Stage 3 (no Anthropic API calls).
        n_synthetic: Number of synthetic pairs to generate.
    """
    if stages is None:
        stages = ["ingest", "preprocess"] + ([] if skip_synthetic else ["synthetic"])

    logger.info("=" * 60)
    logger.info("HireLens Data Pipeline")
    logger.info(f"Stages: {stages}")
    logger.info("=" * 60)

    pipeline_start = time.perf_counter()
    stages_run: list[str] = []

    resume_df = jobs_df = None

    try:
        if "ingest" in stages:
            resume_df, jobs_df = stage_ingest()
            stages_run.append("ingest")

        if "preprocess" in stages:
            stage_preprocess(resume_df=resume_df, jobs_df=jobs_df)
            stages_run.append("preprocess")

        if "synthetic" in stages and not skip_synthetic:
            stage_synthetic(n_pairs=n_synthetic)
            stages_run.append("synthetic")

        total_elapsed = time.perf_counter() - pipeline_start

        try:
            log_pipeline_summary(stages_run, total_elapsed)
        except Exception as e:
            logger.warning(f"MLflow summary logging failed (non-fatal): {e}")

        _verify_outputs()

        logger.info("")
        logger.success(f"Pipeline finished in {total_elapsed:.1f}s. Stages: {stages_run}")

    except Exception as e:
        logger.error(f"Pipeline failed during stage execution: {e}")
        raise


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HireLens data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["ingest", "preprocess", "synthetic"],
        help="Run a single stage (default: all stages)",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic generation (avoids Anthropic API calls)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=500,
        help="Number of synthetic pairs to generate (default: 500)",
    )
    args = parser.parse_args()

    stages = [args.stage] if args.stage else None
    run_pipeline(
        stages=stages,
        skip_synthetic=args.skip_synthetic,
        n_synthetic=args.n_pairs,
    )


if __name__ == "__main__":
    main()
