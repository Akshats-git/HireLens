"""
MLflow experiment configuration and shared helpers.

Usage:
    from registry.config import get_client, EXPERIMENT_NAME
"""

import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "hirelens-resume-matching"
REGISTERED_MODEL_NAME = "hirelens-matcher"

# Metrics that gate staging → production promotion
PRODUCTION_THRESHOLDS = {
    "precision_at_1": 0.85,
    "auc_roc": 0.78,
    "ner_f1": 0.72,
}


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient()


def get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(get_tracking_uri())
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "HireLens", "team": "ml-eng"},
        )
        return exp_id
    return exp.experiment_id
