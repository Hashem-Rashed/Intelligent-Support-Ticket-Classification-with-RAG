"""
Model evaluation metrics and utilities – enhanced with confusion matrix plot.
"""
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
    save_report_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model performance with weighted and macro averages.
    Optionally saves classification report to CSV.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    logger.info("Classification Report:\n" + classification_report(y_true, y_pred, zero_division=0))

    if save_report_path:
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(save_report_path)
        logger.info(f"Classification report saved to {save_report_path}")

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[list] = None) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> None:
    """Plot and optionally save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.show()


class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self, model, labels: Optional[list] = None):
        self.model = model
        self.labels = labels

    def evaluate(self, X_test, y_test, save_dir: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model on test set and optionally save plots/reports."""
        y_pred = self.model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            # Save classification report
            report_file = save_path / "classification_report.csv"
            evaluate_model(y_test, y_pred, save_report_path=str(report_file))
            # Plot confusion matrix
            plot_confusion_matrix(
                y_test, y_pred,
                labels=self.labels if self.labels else sorted(set(y_test)),
                save_path=str(save_path / "confusion_matrix.png")
            )
        return metrics