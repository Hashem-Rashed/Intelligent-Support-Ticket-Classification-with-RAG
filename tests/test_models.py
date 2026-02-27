"""
Unit tests for model module.
"""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models.baseline.tfidf_logreg import TFIDFLogReg
from src.models.evaluation import evaluate_model


class TestTFIDFLogReg:
    """Test cases for TF-IDF + LogReg model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        texts = [
            "account locked cannot login",
            "forgot password reset issue",
            "payment failed transaction error",
            "shipping delayed package status",
            "product broken defective quality",
        ]
        labels = np.array([0, 0, 1, 2, 1])
        return texts, labels

    def test_model_initialization(self):
        """Test model initialization."""
        model = TFIDFLogReg()
        assert model.max_features == 5000
        assert model.model is not None

    def test_model_training(self, sample_data):
        """Test model training."""
        texts, labels = sample_data
        model = TFIDFLogReg()
        model.fit(texts, labels)
        # Training should complete without errors

    def test_model_prediction(self, sample_data):
        """Test model prediction."""
        texts, labels = sample_data
        model = TFIDFLogReg()
        model.fit(texts, labels)

        predictions = model.predict(texts[:2])
        assert len(predictions) == 2


class TestModelEvaluation:
    """Test cases for model evaluation."""

    def test_evaluate_model(self):
        """Test model evaluation."""
        y_true = np.array([0, 1, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 2, 0])

        metrics = evaluate_model(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] == 1.0
