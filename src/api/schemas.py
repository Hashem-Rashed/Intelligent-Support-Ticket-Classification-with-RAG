"""
Request and response schemas for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ClassifyTicketRequest(BaseModel):
    """Request schema for ticket classification."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    title: str = Field(..., description="Ticket title/subject")
    content: str = Field(..., description="Ticket description content")
    priority: Optional[str] = Field(None, description="Ticket priority level")
    customer_id: Optional[str] = Field(None, description="Customer ID")

    class Config:
        schema_extra = {
            "example": {
                "ticket_id": "TICK-001",
                "title": "Login Issues",
                "content": "Unable to login to the platform",
                "priority": "high",
                "customer_id": "CUST-123",
            }
        }


class ClassifyTicketResponse(BaseModel):
    """Response schema for ticket classification."""

    ticket_id: str = Field(..., description="Ticket ID")
    classification: str = Field(..., description="Predicted ticket category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    suggested_category: Optional[str] = Field(
        None, description="Alternative suggested category"
    )

    class Config:
        schema_extra = {
            "example": {
                "ticket_id": "TICK-001",
                "classification": "Technical Support",
                "confidence": 0.95,
                "suggested_category": "Bug Report",
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Service is running normally",
            }
        }
