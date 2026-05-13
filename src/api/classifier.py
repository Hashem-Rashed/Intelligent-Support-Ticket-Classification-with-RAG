"""
Production-ready ticket classifier with smart fraud detection.
Supports both baseline (TF-IDF) and transformer (DistilBERT) models.
"""
import pickle
import json
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from src.utils.logger import get_logger
from src.preprocessing.text_processing import clean_text

logger = get_logger(__name__)


class ProductionTicketClassifier:
    """
    TF-IDF + Logistic Regression classifier with smart fraud detection.
    """

    THRESHOLDS = {
        "Fraud": 0.60,
        "Technical": 0.50,
        "General Inquiry": 0.60,
        "Customer Support": 0.60,
        "Billing": 0.65,
        "Delivery": 0.65,
        "Account": 0.65,
        "Feature Request": 0.65,
        "Security": 0.65,
    }

    FRAUD_KEYWORDS = [
        "fraud", "scam", "unauthorized", "stolen", "hack", "compromised",
        "did not authorize", "didn't authorize", "false charge", "identity theft",
        "used my credit card", "credit card information", "fake order", "fake orders",
        "unauthorized purchase", "unauthorized transaction", "someone used", "wasn't me",
        "without my permission", "without permission", "accessed from another country",
        "logged in from", "unknown device", "account takeover", "account hacked",
        "someone accessed", "unauthorized access",
        "never approved", "did not approve", "didn't approve", "suspicious payment",
        "payment confirmation", "transaction i never", "i never approved",
        # NEW for refund fraud
        "refund request", "refund requests", "without authorization",
        "refund without", "multiple refunds"
    ]

    SUSPICIOUS_KEYWORDS = [
        "unknown charge", "suspicious", "wasn't me", "not me",
        "someone used", "my card was used", "unrecognized transaction",
        "credit card", "fake", "didn't make",
        "another country", "without permission", "unknown device",
        "payment confirmation", "never approved", "refund request"
    ]

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
            self.is_fitted = True
            self.classes = self.label_encoder.classes_.tolist()

            if 'thresholds' in data:
                self.thresholds = data['thresholds']
                logger.info("Using tuned thresholds from training")
            else:
                self.thresholds = self.THRESHOLDS

            metrics_path = model_dir / "baseline_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded baseline model with accuracy: {self.metrics.get('accuracy', 'N/A')}")
            else:
                logger.info("Loaded baseline model (no metrics file)")

        elif model_type == "transformer":
            from src.models.transformer.bert_finetune import BERTFineTune
            self.transformer_model = BERTFineTune.load(model_dir)
            self.is_fitted = True
            self.classes = self.transformer_model.classes_
            self.thresholds = self.THRESHOLDS
            logger.info("Loaded transformer model")

        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'baseline' or 'transformer'.")

        self._clean = lambda x: clean_text(
            x,
            max_words=8,
            remove_greetings_flag=True,
            is_twitter=False
        )

    def predict(self, text: str, return_details: bool = False) -> Union[str, Tuple[str, float, str, bool]]:
        cleaned = self._clean(text)
        if not cleaned:
            cleaned = "general inquiry"

        text_lower = cleaned.lower()

        # Stage 1: Exact keyword match
        for kw in self.FRAUD_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Fraud keyword match: {kw}")
                if return_details:
                    return "Fraud", 0.95, "keyword_match", False
                return "Fraud"

        # Stage 2: Pattern rules
        # Rule: Unauthorized account access
        if ("account" in text_lower or "profile" in text_lower) and \
           any(term in text_lower for term in ["accessed from", "logged in from", "unknown device", "another country", "without permission", "unauthorized access", "without authorization"]):
            logger.debug("Unauthorized account access pattern → Fraud")
            if return_details:
                return "Fraud", 0.85, "pattern_match", False
            return "Fraud"

        # Rule: Suspicious payment / transaction not approved
        if ("payment" in text_lower or "transaction" in text_lower or "charge" in text_lower) and \
           any(term in text_lower for term in ["never approved", "did not approve", "didn't approve", "suspicious", "not recognize", "did not authorize", "without authorization"]):
            logger.debug("Suspicious unauthorized payment pattern → Fraud")
            if return_details:
                return "Fraud", 0.85, "pattern_match", False
            return "Fraud"

        # Rule: Refund fraud (multiple refund requests without authorization)
        if ("refund" in text_lower) and \
           any(term in text_lower for term in ["without authorization", "unauthorized", "without permission", "multiple", "did not request"]):
            logger.debug("Unauthorized refund request pattern → Fraud")
            if return_details:
                return "Fraud", 0.85, "pattern_match", False
            return "Fraud"

        # Rule: Credit card fraud
        if ("credit card" in text_lower or "card" in text_lower) and \
           any(term in text_lower for term in ["unauthorized", "fake", "used", "stolen", "wasn't me"]):
            logger.debug("Credit card fraud pattern → Fraud")
            if return_details:
                return "Fraud", 0.85, "pattern_match", False
            return "Fraud"

        # Stage 3: Model prediction
        if self.model_type == "baseline":
            X = self.vectorizer.transform([cleaned])
            proba = self.classifier.predict_proba(X)[0]
            pred_encoded = self.classifier.predict(X)[0]
            pred = self.label_encoder.inverse_transform([pred_encoded])[0]
            confidence = max(proba)
        else:
            proba = self.transformer_model.predict_proba([cleaned])[0]
            pred = self.transformer_model.predict([cleaned])[0]
            confidence = max(proba)

        threshold = self.thresholds.get(pred, 0.60)
        needs_review = confidence < threshold

        # Low-confidence fraud fallback
        if pred == "Fraud" and confidence < 0.3:
            for kw in self.SUSPICIOUS_KEYWORDS:
                if kw in text_lower:
                    if return_details:
                        return "Fraud", confidence, "model_low", True
                    return "Fraud"
            if return_details:
                return "General Inquiry", confidence, "fallback", True
            return "General Inquiry"

        if return_details:
            return pred, confidence, "model", needs_review
        return pred

    def predict_batch(self, texts: list, return_details: bool = False) -> list:
        return [self.predict(t, return_details) for t in texts]