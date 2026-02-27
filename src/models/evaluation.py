"""
Model evaluation metrics and utilities.
"""
from typing import Optional, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        labels: Label names

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Log detailed classification report
    logger.info("Classification Report:\n" + classification_report(y_true, y_pred))

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self, model, labels: Optional[list] = None):
        """
        Initialize ModelEvaluator.

        Args:
            model: Model to evaluate
            labels: Label names
        """
        self.model = model
        self.labels = labels

    def evaluate(
        self, X_test, y_test
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        logger.info(f"Evaluation metrics: {metrics}")

        return metrics
