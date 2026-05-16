"""
Production-ready ticket classifiers with optional LLM fallback for low confidence.
"""
import pickle
import json
import os
from pathlib import Path
from typing import Tuple, Union, List, Optional
import numpy as np
from src.utils.logger import get_logger
from src.preprocessing.text_processing import clean_text
from src.rag.llm_fallback import LLMFallback

logger = get_logger(__name__)


class ProductionTicketClassifier:
    """
    Single model classifier (baseline or transformer) with optional LLM fallback.
    """

    DEFAULT_THRESHOLDS = {
        "Account": 0.65,
        "Billing": 0.65,
        "Fraud": 0.65,
        "General Inquiry": 0.60,
        "Technical": 0.60,
    }

    def __init__(
        self,
        model_dir: Union[str, Path],
        model_type: str = "baseline",
        use_llm_fallback: bool = True,
        llm_confidence_threshold: float = 0.65,
    ):
        model_dir = Path(model_dir)
        self.model_type = model_type
        self.use_llm_fallback = use_llm_fallback
        self.llm_confidence_threshold = llm_confidence_threshold
        self.llm = None

        # Load trained model
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

        # Initialize LLM fallback if requested
        if self.use_llm_fallback:
            try:
                self.llm = LLMFallback()
                if not self.llm.is_available():
                    logger.warning("LLM fallback requested but Groq API key not set. Fallback disabled.")
                    self.use_llm_fallback = False
                else:
                    logger.info("LLM fallback enabled for low-confidence predictions.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM fallback: {e}")
                self.use_llm_fallback = False

        self._clean = lambda x: clean_text(
            x,
            max_words=None,
            remove_greetings_flag=True,
            is_twitter=False
        )

    def predict_proba(self, text: str) -> np.ndarray:
        cleaned = self._clean(text)
        if not cleaned:
            cleaned = "general inquiry"
        if self.model_type == "baseline":
            X = self.vectorizer.transform([cleaned])
            return self.classifier.predict_proba(X)[0]
        else:
            return self.transformer_model.predict_proba([cleaned])[0]

    def predict(
        self,
        text: str,
        return_details: bool = False,
        allow_llm_fallback: bool = True,
    ) -> Union[str, Tuple[str, float, bool, Optional[str]]]:
        """
        Predict category for a ticket. Optionally use LLM for low confidence.
        
        Returns:
            If return_details=False: category string
            If return_details=True: (category, confidence, needs_review, model_used)
                model_used: "baseline", "transformer", or "llm"
        """
        cleaned = self._clean(text)
        if not cleaned:
            cleaned = "general inquiry"

        # Get model prediction
        proba = self.predict_proba(text)
        pred_idx = np.argmax(proba)
        category = self.classes[pred_idx]
        confidence = proba[pred_idx]
        threshold = self.thresholds.get(category, 0.65)
        needs_review = confidence < threshold

        model_used = self.model_type

        # LLM fallback for low confidence
        if (
            allow_llm_fallback
            and self.use_llm_fallback
            and (needs_review or confidence < self.llm_confidence_threshold)
        ):
            try:
                llm_category, llm_confidence = self.llm.classify_ticket(cleaned)
                # Only override if LLM confidence is higher or model was very uncertain
                if llm_confidence > confidence + 0.1:
                    category = llm_category
                    confidence = llm_confidence
                    needs_review = False
                    model_used = "llm"
                    logger.info(f"LLM fallback used: {category} (conf={llm_confidence:.2f})")
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")

        if return_details:
            return category, confidence, needs_review, model_used
        return category

    def predict_batch(self, texts: List[str], return_details: bool = False) -> List:
        return [self.predict(t, return_details) for t in texts]


class EnsembleTicketClassifier:
    """
    Ensemble classifier that averages probabilities from baseline and transformer.
    Falls back to single model if one is missing.
    Also supports LLM fallback for low confidence.
    """
    def __init__(
        self,
        baseline_dir: Union[str, Path],
        transformer_dir: Union[str, Path],
        use_llm_fallback: bool = True,
        llm_confidence_threshold: float = 0.65,
    ):
        self.baseline = None
        self.transformer = None
        self.classes = None
        self.use_llm_fallback = use_llm_fallback
        self.llm_confidence_threshold = llm_confidence_threshold
        self.llm = None

        try:
            self.baseline = ProductionTicketClassifier(baseline_dir, model_type="baseline", use_llm_fallback=False)
            logger.info("Baseline model loaded for ensemble.")
        except Exception as e:
            logger.warning(f"Could not load baseline model: {e}")

        try:
            self.transformer = ProductionTicketClassifier(transformer_dir, model_type="transformer", use_llm_fallback=False)
            logger.info("Transformer model loaded for ensemble.")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")

        if self.baseline is None and self.transformer is None:
            raise RuntimeError("No models available for ensemble.")

        # Use classes from whichever model is available
        if self.baseline:
            self.classes = self.baseline.classes
        else:
            self.classes = self.transformer.classes

        self.thresholds = ProductionTicketClassifier.DEFAULT_THRESHOLDS

        # Initialize LLM fallback
        if self.use_llm_fallback:
            try:
                self.llm = LLMFallback()
                if not self.llm.is_available():
                    logger.warning("LLM fallback requested but Groq API key not set.")
                    self.use_llm_fallback = False
                else:
                    logger.info("LLM fallback enabled for ensemble low-confidence predictions.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM fallback: {e}")
                self.use_llm_fallback = False

    def predict_proba(self, text: str) -> np.ndarray:
        probs = []
        if self.baseline:
            probs.append(self.baseline.predict_proba(text))
        if self.transformer:
            probs.append(self.transformer.predict_proba(text))
        if not probs:
            raise RuntimeError("No model available")
        return np.mean(probs, axis=0)

    def predict(
        self,
        text: str,
        return_details: bool = False,
        allow_llm_fallback: bool = True,
    ) -> Union[str, Tuple[str, float, bool, Optional[str]]]:
        proba = self.predict_proba(text)
        pred_idx = np.argmax(proba)
        category = self.classes[pred_idx]
        confidence = proba[pred_idx]
        threshold = self.thresholds.get(category, 0.65)
        needs_review = confidence < threshold
        model_used = "ensemble"

        # LLM fallback for low confidence
        if (
            allow_llm_fallback
            and self.use_llm_fallback
            and (needs_review or confidence < self.llm_confidence_threshold)
        ):
            try:
                cleaned = self.baseline._clean(text) if self.baseline else text
                llm_category, llm_confidence = self.llm.classify_ticket(cleaned)
                if llm_confidence > confidence + 0.1:
                    category = llm_category
                    confidence = llm_confidence
                    needs_review = False
                    model_used = "llm"
                    logger.info(f"LLM fallback used: {category} (conf={llm_confidence:.2f})")
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")

        if return_details:
            return category, confidence, needs_review, model_used
        return category

    def predict_batch(self, texts: List[str], return_details: bool = False) -> List:
        return [self.predict(t, return_details) for t in texts]