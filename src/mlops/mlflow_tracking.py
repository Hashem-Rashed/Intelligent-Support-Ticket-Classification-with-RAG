"""
MLflow experiment tracking utilities.
"""
from typing import Dict, Optional
import mlflow
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLFlowTracker:
    """MLflow experiment tracking wrapper."""

    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize MLFlowTracker.

        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # Set or create experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Set MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting MLflow experiment: {str(e)}")

    def log_params(self, params: Dict) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Epoch or step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """
        Log model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path to save model
        """
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Logged model to MLflow: {artifact_path}")

    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
        """
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
