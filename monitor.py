"""
HireLens — Automated retraining monitor.

Reads the latest evaluation_report.json produced by `dvc repro evaluate`,
compares key metrics against thresholds defined in params.yaml, and triggers
a DVC repro pipeline run (train + evaluate) if any metric drops below its
threshold.

Usage:
    python monitor.py                    # one-shot check
    python monitor.py --daemon           # loop every 24 hours
    python monitor.py --force-retrain    # trigger immediately (testing)

CloudWatch integration:
    Set AWS_DEFAULT_REGION + AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY and
    the script will emit custom metrics to the 'HireLens/ML' namespace.
    Without credentials, alerts are written to logs/monitor_alerts.log only.
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

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_REPORT = PROJECT_ROOT / "logs" / "evaluation_report.json"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
ALERT_LOG = PROJECT_ROOT / "logs" / "monitor_alerts.log"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)
logger.add(ALERT_LOG, level="WARNING", rotation="20 MB", retention="60 days")


# ── Params ────────────────────────────────────────────────────────────────────


def _load_thresholds() -> dict:
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    return params.get("monitor", {}).get(
        "retraining_threshold",
        {
            "precision_at_1": 0.85,
            "auc_roc": 0.78,
        },
    )


def _check_interval_hours() -> float:
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    return params.get("monitor", {}).get("check_interval_hours", 24)


# ── Report reading ────────────────────────────────────────────────────────────


def _load_eval_report() -> dict:
    if not EVAL_REPORT.exists():
        raise FileNotFoundError(f"Eval report not found: {EVAL_REPORT}")
    with open(EVAL_REPORT) as f:
        return json.load(f)


def _extract_key_metrics(report: dict) -> dict:
    retrieval = report.get("retrieval", {})
    ner = report.get("ner", {})
    return {
        "precision_at_1": retrieval.get("precision_at_1", 0.0),
        "ndcg_at_10": retrieval.get("ndcg_at_10", 0.0),
        "auc_roc": retrieval.get("auc_roc", 0.0),
        "mrr": retrieval.get("mrr", 0.0),
        "ner_f1": ner.get("f1", 0.0),
    }


# ── Threshold check ───────────────────────────────────────────────────────────


def check_metrics(metrics: dict, thresholds: dict) -> list[tuple[str, float, float]]:
    """Return list of (metric, actual, threshold) for every failing metric."""
    return [
        (name, metrics.get(name, 0.0), thresh)
        for name, thresh in thresholds.items()
        if metrics.get(name, 0.0) < thresh
    ]


# ── CloudWatch ────────────────────────────────────────────────────────────────


def _emit_cloudwatch(metrics: dict, failures: list) -> None:
    try:
        import boto3

        cw = boto3.client("cloudwatch")
        metric_data = [
            {
                "MetricName": name.replace("_", "-"),
                "Value": value,
                "Unit": "None",
                "Dimensions": [{"Name": "Model", "Value": "hirelens-matcher"}],
            }
            for name, value in metrics.items()
        ]
        if failures:
            metric_data.append(
                {
                    "MetricName": "RetrainingTriggered",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Model", "Value": "hirelens-matcher"}],
                }
            )
        cw.put_metric_data(Namespace="HireLens/ML", MetricData=metric_data)
        logger.info("Metrics emitted to CloudWatch (HireLens/ML namespace)")
    except ImportError:
        logger.debug("boto3 not installed — CloudWatch skipped")
    except Exception as exc:
        logger.warning(f"CloudWatch emit failed: {exc}")


# ── Retraining ────────────────────────────────────────────────────────────────


def trigger_retraining() -> bool:
    """Run `dvc repro train evaluate` and return True on success."""
    logger.warning("Triggering automated retraining via DVC...")
    cmd = ["dvc", "repro", "train", "evaluate", "--force"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if result.returncode == 0:
        logger.success("Retraining completed successfully.")
        return True
    else:
        logger.error(f"Retraining failed (exit code {result.returncode}).")
        return False


# ── Core check cycle ──────────────────────────────────────────────────────────


def run_check(force_retrain: bool = False) -> bool:
    """
    Single monitoring cycle. Returns True if retraining was triggered.
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(f"[{ts}] Running monitoring check...")

    try:
        report = _load_eval_report()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return False

    metrics = _extract_key_metrics(report)
    thresholds = _load_thresholds()

    logger.info("Current metrics vs thresholds:")
    for name, thresh in thresholds.items():
        actual = metrics.get(name, 0.0)
        status = "✓" if actual >= thresh else "✗"
        logger.info(f"  {status} {name}: {actual:.4f}  (threshold ≥ {thresh})")

    failures = check_metrics(metrics, thresholds)
    _emit_cloudwatch(metrics, failures)

    if not failures and not force_retrain:
        logger.success("All metrics within thresholds — no retraining needed.")
        return False

    if failures:
        msg = "Metric regression detected — triggering retraining: " + ", ".join(
            f"{n}={a:.4f}<{t}" for n, a, t in failures
        )
        logger.warning(msg)

    return trigger_retraining()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="HireLens retraining monitor")
    parser.add_argument(
        "--daemon", action="store_true", help="Run as a daemon — check every N hours"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Trigger retraining regardless of metric values",
    )
    args = parser.parse_args()

    if args.daemon:
        interval = _check_interval_hours()
        logger.info(f"Daemon mode: checking every {interval}h")
        while True:
            run_check(force_retrain=args.force_retrain)
            logger.info(f"Next check in {interval}h — sleeping...")
            time.sleep(interval * 3600)
    else:
        triggered = run_check(force_retrain=args.force_retrain)
        sys.exit(0 if not triggered else 0)


if __name__ == "__main__":
    main()
