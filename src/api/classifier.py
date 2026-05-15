"""
Production-ready ticket classifiers.
Supports baseline (TF‑IDF + LR), transformer (DistilBERT), and ensemble (average probabilities).
All rely solely on trained models – no keyword rules.
"""
import pickle
import json
from pathlib import Path
from typing import Tuple, Union, List, Optional
import numpy as np
from src.utils.logger import get_logger
from src.preprocessing.text_processing import clean_text

logger = get_logger(__name__)


class ProductionTicketClassifier:
    """
    Single model classifier (baseline or transformer).
    """

    DEFAULT_THRESHOLDS = {
        "Account": 0.65,
        "Billing": 0.65,
        "Fraud": 0.65,
        "General Inquiry": 0.60,
        "Technical": 0.60,
    }

    def __init__(self, model_dir: Union[str, Path], model_type: str = "baseline"):
        model_dir = Path(model_dir)
        self.model_type = model_type

        if model_type == "baseline":
            model_path = model_dir / "tfidf_logreg_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Baseline model not found at {model_path}")
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.label_encoder = data["label_encoder"]
            self.classes = self.label_encoder.classes_.tolist()
            self.thresholds = data.get("thresholds", self.DEFAULT_THRESHOLDS)
            self.is_fitted = True
            logger.info(f"Loaded baseline model with classes: {self.classes}")
        elif model_type == "transformer":
            from src.models.transformer.bert_finetune import BERTFineTune
            self.transformer_model = BERTFineTune.load(model_dir)
            self.classes = self.transformer_model.classes_
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()
            if hasattr(self.transformer_model, "thresholds"):
                self.thresholds.update(self.transformer_model.thresholds)
            self.is_fitted = True
            logger.info(f"Loaded transformer model with classes: {self.classes}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._clean = lambda x: clean_text(
            x,
            max_words=None,
            remove_greetings_flag=True,
            is_twitter=False
        )

    def predict_proba(self, text: str) -> np.ndarray:
        """Return probability vector for the given text."""
        cleaned = self._clean(text)
        if not cleaned:
            cleaned = "general inquiry"
        if self.model_type == "baseline":
            X = self.vectorizer.transform([cleaned])
            return self.classifier.predict_proba(X)[0]
        else:
            return self.transformer_model.predict_proba([cleaned])[0]

    def predict(self, text: str, return_details: bool = False) -> Union[str, Tuple[str, float, bool]]:
        cleaned = self._clean(text)
        if not cleaned:
            cleaned = "general inquiry"

        proba = self.predict_proba(text)
        pred_idx = np.argmax(proba)
        pred = self.classes[pred_idx]
        confidence = proba[pred_idx]

        threshold = self.thresholds.get(pred, 0.60)
        needs_review = confidence < threshold

        if return_details:
            return pred, confidence, needs_review
        return pred

    def predict_batch(self, texts: List[str], return_details: bool = False) -> List:
        return [self.predict(t, return_details) for t in texts]


class EnsembleTicketClassifier:
    """
    Ensemble classifier that averages probabilities from baseline and transformer.
    Falls back to single model if one is missing.
    """
    def __init__(self, baseline_dir: Union[str, Path], transformer_dir: Union[str, Path]):
        self.baseline = None
        self.transformer = None
        self.classes = None

        try:
            self.baseline = ProductionTicketClassifier(baseline_dir, model_type="baseline")
            logger.info("Baseline model loaded for ensemble.")
        except Exception as e:
            logger.warning(f"Could not load baseline model: {e}")

        try:
            self.transformer = ProductionTicketClassifier(transformer_dir, model_type="transformer")
            logger.info("Transformer model loaded for ensemble.")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")

        if self.baseline is None and self.transformer is None:
            raise RuntimeError("No models available for ensemble.")

        # Use classes from whichever model is available (they should be identical)
        if self.baseline:
            self.classes = self.baseline.classes
        else:
            self.classes = self.transformer.classes

        # Define thresholds (could be same as single model defaults)
        self.thresholds = ProductionTicketClassifier.DEFAULT_THRESHOLDS

    def predict_proba(self, text: str) -> np.ndarray:
        probs = []
        if self.baseline:
            probs.append(self.baseline.predict_proba(text))
        if self.transformer:
            probs.append(self.transformer.predict_proba(text))
        if not probs:
            raise RuntimeError("No model available")
        # Average probabilities
        return np.mean(probs, axis=0)

    def predict(self, text: str, return_details: bool = False) -> Union[str, Tuple[str, float, bool]]:
        proba = self.predict_proba(text)
        pred_idx = np.argmax(proba)
        pred = self.classes[pred_idx]
        confidence = proba[pred_idx]

        threshold = self.thresholds.get(pred, 0.60)
        needs_review = confidence < threshold

        if return_details:
            return pred, confidence, needs_review
        return pred

    def predict_batch(self, texts: List[str], return_details: bool = False) -> List:
        return [self.predict(t, return_details) for t in texts]