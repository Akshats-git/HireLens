"""
MLflow tracking configuration shared by the registry commands.

Values come from configs/config.yaml; only the tracking URI is environment-driven
so the same code works locally and inside Docker.

Usage:
    from registry.config import get_client, EXPERIMENT_NAME
"""

import os
from functools import lru_cache
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DEFAULT_TRACKING_URI = "http://localhost:5000"


@lru_cache(maxsize=1)
def _config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


EXPERIMENT_NAME: str = _config()["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME: str = _config()["mlflow"]["registered_model_name"]

# Metrics a run must clear before it is aliased as 'production'.
PROMOTION_THRESHOLDS: dict[str, float] = _config()["evaluation"]["promotion_thresholds"]


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI, honouring MLFLOW_TRACKING_URI."""
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_client() -> MlflowClient:
    """Return an MlflowClient bound to the configured tracking server."""
    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient()


def get_or_create_experiment() -> str:
    """Return the HireLens experiment ID, creating the experiment if needed."""
    mlflow.set_tracking_uri(get_tracking_uri())
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        EXPERIMENT_NAME, tags=_config()["mlflow"]["run_tags"]
    )
