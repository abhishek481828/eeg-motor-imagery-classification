"""MLflow experiment tracking logger with git commit and environment metadata logging."""

import os
import platform
import subprocess
import sys
from typing import Any

import mlflow


def get_git_commit_hash() -> str:
    """Return short git commit hash or 'unknown' if not in git repo."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "unknown"


def init_mlflow(
    experiment_name: str = "eeg_motor_imagery_research",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> None:
    """Initialize MLflow tracking URI and active experiment."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_environment_metadata() -> None:
    """Log Python version, OS platform, and Git commit hash to MLflow."""
    mlflow.log_param("git_commit_hash", get_git_commit_hash())
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("platform", platform.platform())


def log_experiment_params(params: dict[str, Any], prefix: str = "") -> None:
    """Log flattened configuration parameters to active MLflow run."""
    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_experiment_params(value, prefix=param_name)
        else:
            mlflow.log_param(param_name, str(value))


def log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
    """Log training and validation metrics for a single epoch."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=epoch)
