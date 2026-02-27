"""
TF-IDF and Logistic Regression baseline model.
"""
from typing import Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFIDFLogReg:
    """TF-IDF + Logistic Regression baseline model."""

    def __init__(
        self,
        max_features: int = 5000,
        max_df: float = 0.8,
        min_df: int = 2,
        C: float = 1.0,
    ):
        """
        Initialize TFIDFLogReg model.

        Args:
            max_features: Maximum number of features
            max_df: Maximum document frequency
            min_df: Minimum document frequency
            C: Inverse of regularization strength
        """
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.C = C

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                max_df=max_df,
                min_df=min_df,
                stop_words="english"
            )),
            ("logreg", LogisticRegression(C=C, max_iter=1000))
        ])

    def fit(self, X: list, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: List of texts
            y: Array of labels
        """
        logger.info("Training TF-IDF + LogReg model")
        self.model.fit(X, y)
        logger.info("Training complete")

    def predict(self, X: list) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: List of texts

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: list) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: List of texts

        Returns:
            Prediction probabilities
        """
        return self.model.named_steps["logreg"].predict_proba(
            self.model.named_steps["tfidf"].transform(X)
        )

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TFIDFLogReg":
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model instance
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        instance = cls()
        instance.model = model
        logger.info(f"Model loaded from {filepath}")
        return instance
