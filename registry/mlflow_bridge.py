"""
registry/mlflow_bridge.py
--------------------------
Thin wrapper that logs a training run to MLflow and registers the
best model in the MLflow Model Registry — using the same artifact
paths produced by train.py so the two registries stay in sync.

This is a bridge, not a replacement. The file-system registry in
model_registry.py remains the source of truth for deployment;
MLflow is the experiment-tracking and audit layer.

Usage:
    python -m registry.mlflow_bridge \
        --model-path artifacts/model.pkl \
        --preprocessor-path artifacts/preprocessor.pkl \
        --run-log artifacts/run_log.json
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_to_mlflow(
    model_path: str | Path,
    preprocessor_path: str | Path,
    run_log_path: str | Path,
    experiment_name: str = "ml-versioning-demo",
    registry_name: str = "fraud-classifier",
) -> str:
    """
    Log a training run to MLflow and register the model.
    Returns the MLflow run ID.

    Requires:
        pip install mlflow scikit-learn
    """
    try:
        import mlflow
        import mlflow.sklearn
        import joblib
    except ImportError:
        raise ImportError("Run: pip install mlflow scikit-learn")

    with open(run_log_path) as f:
        run_log = json.load(f)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_log.get("run_id", "unnamed")) as run:
        # Log all hyperparameters from the winner config
        winner_cfg = run_log.get("winner_config", {})
        for k, v in winner_cfg.items():
            if k != "model_class":
                mlflow.log_param(k, v)
        mlflow.log_param("model_class", winner_cfg.get("model_class", "unknown"))
        mlflow.log_param("data_window", run_log.get("data_window", "unknown"))
        mlflow.log_param("git_commit", run_log.get("git_commit", "unknown"))

        # Log validation metrics
        val_metrics = run_log.get("val_metrics", {})
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v)

        # Log CV stats
        mlflow.log_metric("cv_roc_auc_mean", run_log.get("cv_roc_auc_mean", 0))
        mlflow.log_metric("cv_roc_auc_std", run_log.get("cv_roc_auc_std", 0))

        # Log the model as an MLflow artifact
        model = joblib.load(model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registry_name,
        )

        # Log the preprocessor as a plain artifact for traceability
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        mlflow.log_artifact(str(run_log_path), artifact_path="run_log")

        run_id = run.info.run_id
        logger.info(
            "MLflow run logged | run_id=%s | experiment=%s",
            run_id, experiment_name
        )

    return run_id


def transition_model_stage(
    registry_name: str,
    version: int,
    stage: str,  # "Staging", "Production", "Archived"
) -> None:
    """
    Transition a registered model version in MLflow to a new stage.

    Call this after your evaluation gate passes:
        transition_model_stage("fraud-classifier", 3, "Production")
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError("Run: pip install mlflow")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info(
        "Transitioned %s v%s → %s",
        registry_name, version, stage
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preprocessor-path", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--experiment", default="ml-versioning-demo")
    parser.add_argument("--registry-name", default="fraud-classifier")
    args = parser.parse_args()

    run_id = log_to_mlflow(
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        run_log_path=args.run_log,
        experiment_name=args.experiment,
        registry_name=args.registry_name,
    )
    print(f"MLflow run ID: {run_id}")
