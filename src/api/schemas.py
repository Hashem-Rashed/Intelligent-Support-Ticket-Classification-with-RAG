"""
Request and response schemas for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ClassifyRequest(BaseModel):
    """Request schema for ticket classification."""
    text: str = Field(..., description="Ticket description text", min_length=1)
    model_type: Optional[str] = Field("ensemble", description="Model to use: 'baseline', 'transformer', or 'ensemble'")
    return_details: Optional[bool] = Field(True, description="Return confidence and needs_review flag")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Someone stole my credit card and made unauthorized purchases",
                "model_type": "ensemble",
                "return_details": True
            }
        }


class ClassifyResponse(BaseModel):
    """Response schema for ticket classification."""
    category: str = Field(..., description="Predicted category (Account, Billing, Fraud, General Inquiry, Technical)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    needs_review: Optional[bool] = Field(None, description="Whether human review is recommended")
    model_used: Optional[str] = Field(None, description="Which model produced the prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Fraud",
                "confidence": 0.97,
                "needs_review": False,
                "model_used": "ensemble"
            }
        }


class BatchClassifyRequest(BaseModel):
    """Request schema for batch classification."""
    tickets: List[ClassifyRequest] = Field(..., description="List of tickets to classify")


class BatchClassifyResponse(BaseModel):
    """Response schema for batch classification."""
    results: List[ClassifyResponse]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    models_available: List[str] = Field(..., description="List of loaded models")