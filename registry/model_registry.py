"""
MLflow model registry helper — promotes models from Staging to Production.

MLflow 2.9+ uses aliases instead of lifecycle stages (Staging/Production).
This script uses set_registered_model_alias / get_model_version_by_alias.

Usage:
    python -m registry.model_registry                      # auto-find best training run & promote
    python -m registry.model_registry --run-id <run_id>   # register a specific run
    python -m registry.model_registry --dry-run            # check thresholds, no promotion
    python -m registry.model_registry --list               # list all versions + aliases
    python -m registry.model_registry --promote-version 2  # directly alias a version as 'production'
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_REPORT = PROJECT_ROOT / "logs" / "evaluation_report.json"
MODEL_DIR = PROJECT_ROOT / "models" / "fine_tuned" / "hirelens_matcher"

from registry.config import (
    EXPERIMENT_NAME,
    PRODUCTION_THRESHOLDS,
    REGISTERED_MODEL_NAME,
    get_client,
    get_or_create_experiment,
    get_tracking_uri,
)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)

# Aliases used instead of deprecated lifecycle stages
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_run_metrics(client: MlflowClient, run_id: str) -> dict:
    return client.get_run(run_id).data.metrics


def _passes_thresholds(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    for metric, threshold in PRODUCTION_THRESHOLDS.items():
        actual = metrics.get(metric)
        if actual is None:
            failures.append(f"  {metric}: MISSING (required ≥ {threshold})")
        elif actual < threshold:
            failures.append(f"  {metric}: {actual:.4f} < {threshold} (threshold)")
    return len(failures) == 0, failures


def _ensure_registered_model(client: MlflowClient) -> None:
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description="HireLens fine-tuned sentence-transformer for resume–JD matching",
            tags={"framework": "sentence-transformers", "task": "semantic-similarity"},
        )
        logger.info(f"Created registered model: {REGISTERED_MODEL_NAME}")


# ── Core actions ──────────────────────────────────────────────────────────────


def register_run(client: MlflowClient, run_id: str) -> str:
    """Register a run's model artifact and alias it as 'staging'. Returns version string."""
    model_uri = f"runs:/{run_id}/model"
    version_info = mlflow.register_model(
        model_uri=model_uri, name=REGISTERED_MODEL_NAME
    )
    version = version_info.version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_STAGING, version)
    logger.info(f"Registered run {run_id} → version {version} (alias: {ALIAS_STAGING})")
    return version


def promote_to_production(client: MlflowClient, version: str) -> None:
    """Alias a version as 'production' (moves the alias, old version keeps its other aliases)."""
    # Remove production alias from whatever currently holds it
    try:
        current = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, ALIAS_PRODUCTION
        )
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
        logger.info(
            f"Removed '{ALIAS_PRODUCTION}' alias from version {current.version}"
        )
    except mlflow.exceptions.MlflowException:
        pass  # No current production version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION, version)
    logger.success(f"Version {version} is now aliased as '{ALIAS_PRODUCTION}' ✓")


def list_versions(client: MlflowClient) -> None:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        logger.info("No registered versions found.")
        return

    # model.aliases is already a dict[alias_str → version_str]
    try:
        model = client.get_registered_model(REGISTERED_MODEL_NAME)
        alias_map: dict[str, str] = dict(model.aliases)  # {'production': '2', ...}
    except Exception:
        alias_map = {}

    print(f"\n{'Ver':<5} {'Aliases':<22} {'Run ID':<32} {'Created':<20}")
    print("-" * 82)
    for v in sorted(versions, key=lambda x: int(x.version)):
        ts = datetime.datetime.fromtimestamp(v.creation_timestamp // 1000).strftime(
            "%Y-%m-%d %H:%M"
        )
        aliases = [a for a, ver in alias_map.items() if ver == str(v.version)]
        alias_str = ", ".join(aliases) if aliases else "-"
        print(f"{v.version:<5} {alias_str:<22} {v.run_id:<32} {ts}")
    print()


def seed_run_from_eval_report() -> str:
    """
    Create an MLflow run from the existing evaluation_report.json + local model dir.
    Use this when the original training run never reached the MLflow server.
    Returns the new run_id.
    """
    if not EVAL_REPORT.exists():
        logger.error(f"Eval report not found: {EVAL_REPORT}")
        sys.exit(1)
    if not MODEL_DIR.exists():
        logger.error(f"Model directory not found: {MODEL_DIR}")
        sys.exit(1)

    with open(EVAL_REPORT) as f:
        report = json.load(f)

    retrieval = report.get("retrieval", {})
    ner = report.get("ner", {})

    flat_metrics = {
        "precision_at_1": retrieval.get("precision_at_1", 0.0),
        "precision_at_3": retrieval.get("precision_at_3", 0.0),
        "precision_at_5": retrieval.get("precision_at_5", 0.0),
        "ndcg_at_10": retrieval.get("ndcg_at_10", 0.0),
        "auc_roc": retrieval.get("auc_roc", 0.0),
        "mrr": retrieval.get("mrr", 0.0),
        "pearson_cosine": retrieval.get("pearson_cosine", 0.0),
        "spearman_cosine": retrieval.get("spearman_cosine", 0.0),
        "ner_f1": ner.get("f1", 0.0),
        "ner_precision": ner.get("precision", 0.0),
        "ner_recall": ner.get("recall", 0.0),
    }

    exp_id = get_or_create_experiment()
    with mlflow.start_run(
        experiment_id=exp_id, run_name="seeded-from-eval-report"
    ) as run:
        mlflow.log_metrics(flat_metrics)
        mlflow.log_param("model_path", str(MODEL_DIR))
        mlflow.log_param("source", "seeded-from-existing-eval-report")
        mlflow.set_tag("stage", "fine_tuning")
        mlflow.set_tag("seeded", "true")

        logger.info(f"Logging model artifact from {MODEL_DIR} (may take a moment)...")
        try:
            mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
        except Exception as exc:
            logger.warning(f"Artifact upload skipped (no remote store): {exc}")
            mlflow.log_param("local_model_path", str(MODEL_DIR))

        run_id = run.info.run_id

    logger.success(f"Seeded MLflow run: {run_id}")
    logger.info("Metrics logged:")
    for k, v in flat_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    return run_id


def find_best_training_run(client: MlflowClient) -> str | None:
    """
    Return the run_id of the best finished fine-tuning run.
    Filters for runs that have 'precision_at_1' logged (excludes pipeline-summary runs).
    """
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED' AND metrics.precision_at_1 > 0",
        order_by=["metrics.precision_at_1 DESC"],
        max_results=1,
    )
    if not runs:
        logger.warning(
            "No finished training runs with 'precision_at_1' found.\n"
            "  → Run training first: dvc repro train\n"
            "  → Or specify a run manually: python -m registry.model_registry --run-id <id>"
        )
        return None
    return runs[0].info.run_id


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HireLens MLflow model registry manager"
    )
    parser.add_argument("--run-id", default=None, help="MLflow run ID to register")
    parser.add_argument(
        "--list", action="store_true", help="List all registered model versions"
    )
    parser.add_argument(
        "--promote-version",
        default=None,
        help="Directly promote a version number to production alias",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check thresholds but do not register or promote",
    )
    parser.add_argument(
        "--seed-from-eval",
        action="store_true",
        help="Create an MLflow run from logs/evaluation_report.json + local model, "
        "then register and promote if thresholds pass",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(get_tracking_uri())
    client = get_client()
    get_or_create_experiment()
    _ensure_registered_model(client)

    if args.list:
        list_versions(client)
        return

    if args.seed_from_eval:
        args.run_id = seed_run_from_eval_report()

    if args.promote_version:
        promote_to_production(client, args.promote_version)
        return

    # Auto-find or use specified run
    run_id = args.run_id or find_best_training_run(client)
    if run_id is None:
        sys.exit(1)

    metrics = _get_run_metrics(client, run_id)
    logger.info(f"Run {run_id} metrics:")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v:.4f}")

    passed, failures = _passes_thresholds(metrics)

    if not passed:
        logger.warning("Run does NOT meet production thresholds:")
        for f in failures:
            logger.warning(f)
        if not args.dry_run:
            version = register_run(client, run_id)
            logger.info(
                f"Registered to '{ALIAS_STAGING}' (version {version}) for manual review."
            )
        sys.exit(0)

    logger.success("All thresholds passed ✓")

    if args.dry_run:
        logger.info("Dry run — skipping promotion.")
        return

    version = register_run(client, run_id)
    promote_to_production(client, version)
    logger.success(f"Model version {version} is now in production.")


if __name__ == "__main__":
    main()
