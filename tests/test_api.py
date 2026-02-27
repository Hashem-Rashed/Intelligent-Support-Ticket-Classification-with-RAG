"""
Unit tests for API module.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import ClassifyTicketRequest


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestClassificationEndpoint:
    """Test cases for ticket classification endpoint."""

    def test_classify_ticket(self, client):
        """Test single ticket classification."""
        request = ClassifyTicketRequest(
            ticket_id="TICK-001",
            title="Login Issue",
            content="Cannot login to system",
        )
        response = client.post("/api/v1/classify", json=request.dict())
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "confidence" in data

    def test_classify_batch(self, client):
        """Test batch ticket classification."""
        requests = [
            ClassifyTicketRequest(
                ticket_id="TICK-001",
                title="Login Issue",
                content="Cannot login",
            ),
            ClassifyTicketRequest(
                ticket_id="TICK-002",
                title="Payment Error",
                content="Payment failed",
            ),
        ]
        response = client.post("/api/v1/classify/batch", json=[r.dict() for r in requests])
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
