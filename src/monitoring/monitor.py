"""
Automated retraining monitor.

Reads the latest evaluation report produced by `dvc repro evaluate`, compares
key metrics against the thresholds in configs/config.yaml, and triggers a DVC
repro run (train + evaluate) if any metric has dropped below its threshold.

Usage:
    python -m src.monitoring.monitor                  # one-shot check
    python -m src.monitoring.monitor --daemon         # loop on the configured interval
    python -m src.monitoring.monitor --force-retrain  # retrain regardless of metrics

Exit codes:
    0  metrics healthy, or retraining ran successfully
    1  retraining was triggered and failed, or the report could not be read

CloudWatch integration:
    With AWS credentials in the environment, metrics are emitted to the
    'HireLens/ML' namespace. Without them, alerts go to logs/monitor_alerts.log.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
EVAL_REPORT = PROJECT_ROOT / "logs" / "evaluation_report.json"
ALERT_LOG = PROJECT_ROOT / "logs" / "monitor_alerts.log"

CLOUDWATCH_NAMESPACE = "HireLens/ML"
CLOUDWATCH_DIMENSIONS = [{"Name": "Model", "Value": "hirelens-matcher"}]


# ── Configuration ─────────────────────────────────────────────────────────────


def _load_monitor_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config.get("monitor", {})


# ── Evaluation report ─────────────────────────────────────────────────────────


def load_eval_report() -> dict:
    """Read the evaluation report written by the `evaluate` DVC stage."""
    if not EVAL_REPORT.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {EVAL_REPORT}. Run `dvc repro evaluate` first."
        )
    with open(EVAL_REPORT) as f:
        return json.load(f)


def extract_key_metrics(report: dict) -> dict[str, float]:
    """Flatten the nested evaluation report into the metrics the monitor tracks."""
    retrieval = report.get("retrieval", {})
    ner = report.get("ner", {})
    return {
        "precision_at_1": retrieval.get("precision_at_1", 0.0),
        "ndcg_at_10": retrieval.get("ndcg_at_10", 0.0),
        "auc_roc": retrieval.get("auc_roc", 0.0),
        "mrr": retrieval.get("mrr", 0.0),
        "ner_f1": ner.get("f1", 0.0),
    }


def find_regressions(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Return (metric, actual, threshold) for every metric below its threshold."""
    return [
        (name, metrics.get(name, 0.0), threshold)
        for name, threshold in thresholds.items()
        if metrics.get(name, 0.0) < threshold
    ]


# ── CloudWatch ────────────────────────────────────────────────────────────────


def emit_to_cloudwatch(metrics: dict[str, float], retraining: bool) -> None:
    """Publish current metrics to CloudWatch. Silently skipped without boto3/credentials."""
    try:
        import boto3

        metric_data = [
            {
                "MetricName": name.replace("_", "-"),
                "Value": value,
                "Unit": "None",
                "Dimensions": CLOUDWATCH_DIMENSIONS,
            }
            for name, value in metrics.items()
        ]
        if retraining:
            metric_data.append(
                {
                    "MetricName": "RetrainingTriggered",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": CLOUDWATCH_DIMENSIONS,
                }
            )
        boto3.client("cloudwatch").put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE, MetricData=metric_data
        )
        logger.info(f"Metrics emitted to CloudWatch namespace {CLOUDWATCH_NAMESPACE}")
    except ImportError:
        logger.debug("boto3 not installed — CloudWatch emit skipped.")
    except Exception as exc:
        logger.warning(f"CloudWatch emit failed: {exc}")


# ── Retraining ────────────────────────────────────────────────────────────────


def trigger_retraining() -> bool:
    """Run `dvc repro train evaluate --force`. Returns True on success."""
    logger.warning("Triggering automated retraining via DVC...")
    result = subprocess.run(
        ["dvc", "repro", "train", "evaluate", "--force"], cwd=PROJECT_ROOT
    )
    if result.returncode == 0:
        logger.success("Retraining completed successfully.")
        return True
    logger.error(f"Retraining failed with exit code {result.returncode}.")
    return False


# ── Check cycle ───────────────────────────────────────────────────────────────


def run_check(force_retrain: bool = False) -> bool:
    """
    Run one monitoring cycle.

    Args:
        force_retrain: Retrain even when every metric is within threshold.

    Returns:
        True if no action was needed or retraining succeeded, False otherwise.
    """
    logger.info(
        f"[{datetime.now(timezone.utc).isoformat()}] Running monitoring check..."
    )

    try:
        report = load_eval_report()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return False

    metrics = extract_key_metrics(report)
    thresholds = _load_monitor_config().get("retraining_thresholds", {})

    logger.info("Current metrics vs thresholds:")
    for name, threshold in thresholds.items():
        actual = metrics.get(name, 0.0)
        status = "✓" if actual >= threshold else "✗"
        logger.info(f"  {status} {name}: {actual:.4f}  (threshold ≥ {threshold})")

    regressions = find_regressions(metrics, thresholds)
    emit_to_cloudwatch(metrics, retraining=bool(regressions) or force_retrain)

    if not regressions and not force_retrain:
        logger.success("All metrics within thresholds — no retraining needed.")
        return True

    if regressions:
        logger.warning(
            "Metric regression detected — triggering retraining: "
            + ", ".join(f"{n}={a:.4f}<{t}" for n, a, t in regressions)
        )

    return trigger_retraining()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="HireLens retraining monitor")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, checking on the configured interval",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Trigger retraining regardless of metric values",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}"
    )
    logger.add(ALERT_LOG, level="WARNING", rotation="20 MB", retention="60 days")

    if not args.daemon:
        sys.exit(0 if run_check(force_retrain=args.force_retrain) else 1)

    interval_hours = _load_monitor_config().get("check_interval_hours", 24)
    logger.info(f"Daemon mode: checking every {interval_hours}h")
    while True:
        run_check(force_retrain=args.force_retrain)
        logger.info(f"Next check in {interval_hours}h — sleeping...")
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    main()
