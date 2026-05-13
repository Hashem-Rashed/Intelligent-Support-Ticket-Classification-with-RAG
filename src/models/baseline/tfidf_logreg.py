"""
TF-IDF and Logistic Regression baseline model with label encoder support.
Optimized for better performance.
"""
from typing import Optional, List, Union, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFIDFLogReg:
    """TF-IDF + Logistic Regression baseline model with label encoder."""

    def __init__(
        self,
        max_features: int = 15000,
        max_df: float = 0.6,
        min_df: int = 3,
        ngram_range: tuple = (1, 3),
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        solver: str = "lbfgs",
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver=solver,
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.is_fitted = False
        # NEW: per-class thresholds for high-precision predictions
        self.thresholds: Dict[str, float] = {}

    def fit(self, X: List[str], y: Union[List[str], np.ndarray]) -> None:
        """Train the model."""
        logger.info("Training TF-IDF + LogReg model")
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_.tolist()
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y_encoded)
        self.is_fitted = True
        # Default thresholds = 0.5 for all classes
        self.thresholds = {cls: 0.5 for cls in self.classes_}
        logger.info(f"Training complete. Classes: {self.classes_}")

    def predict(self, X: List[str], apply_thresholds: bool = True) -> np.ndarray:
        """
        Predict categories (strings).
        If apply_thresholds=True, uses per-class thresholds from self.thresholds.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        if apply_thresholds and self.thresholds:
            return self.predict_with_threshold(X, self.thresholds)
        X_tfidf = self.vectorizer.transform(X)
        y_encoded = self.classifier.predict(X_tfidf)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_tfidf)

    def predict_with_threshold(self, X: List[str], thresholds: Dict[str, float]) -> np.ndarray:
        """
        Predict using per-class thresholds.
        If the highest probability class is below its threshold, fall back to second best
        (or to 'General Inquiry' if none meets 0.5).
        """
        proba = self.predict_proba(X)
        y_encoded = []
        class_indices = {cls: idx for idx, cls in enumerate(self.classes_)}
        fallback_idx = class_indices.get('General Inquiry', 0)

        for probs in proba:
            # Sort classes by probability descending
            sorted_indices = np.argsort(probs)[::-1]
            chosen_idx = None
            for idx in sorted_indices:
                cls = self.classes_[idx]
                thr = thresholds.get(cls, 0.5)
                if probs[idx] >= thr:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                # No class meets its threshold → fallback
                # First try the highest probability class if it's > 0.3
                if probs[sorted_indices[0]] >= 0.3:
                    chosen_idx = sorted_indices[0]
                else:
                    chosen_idx = fallback_idx
            y_encoded.append(chosen_idx)

        return self.label_encoder.inverse_transform(np.array(y_encoded))

    def predict_with_confidence(self, X: List[str]) -> List[tuple]:
        """Return (predicted_category, confidence)."""
        proba = self.predict_proba(X)
        preds = self.predict(X, apply_thresholds=True)
        confidences = proba.max(axis=1)
        return list(zip(preds, confidences))

    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set per-class decision thresholds."""
        self.thresholds = thresholds
        logger.info(f"Thresholds set: {thresholds}")

    def save(self, filepath: str) -> None:
        """Save full pipeline including label encoder and thresholds."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "thresholds": self.thresholds,
                "params": {
                    "max_features": self.max_features,
                    "max_df": self.max_df,
                    "min_df": self.min_df,
                    "ngram_range": self.ngram_range,
                    "C": self.C,
                    "class_weight": self.class_weight,
                }
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TFIDFLogReg":
        """Load full pipeline."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            max_features=data["params"]["max_features"],
            max_df=data["params"]["max_df"],
            min_df=data["params"]["min_df"],
            ngram_range=data["params"]["ngram_range"],
            C=data["params"]["C"],
            class_weight=data["params"]["class_weight"],
        )
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.label_encoder = data["label_encoder"]
        instance.thresholds = data.get("thresholds", {cls: 0.5 for cls in instance.label_encoder.classes_})
        instance.classes_ = instance.label_encoder.classes_.tolist()
        instance.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return instance