"""
Request and response schemas for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Ticket description text", min_length=1)
    model_type: Optional[str] = Field("ensemble", description="Model to use: 'baseline', 'transformer', or 'ensemble'")
    return_details: Optional[bool] = Field(True, description="Return confidence and needs_review flag")
    confidence_threshold: Optional[float] = Field(0.65, description="Minimum confidence to avoid review", ge=0.0, le=1.0)
    use_llm_fallback: Optional[bool] = Field(True, description="Use Groq LLM when confidence is low")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Someone stole my credit card and made unauthorized purchases",
                "model_type": "ensemble",
                "return_details": True,
                "confidence_threshold": 0.65,
                "use_llm_fallback": True
            }
        }


class RagExplainRequest(ClassifyRequest):
    top_k: Optional[int] = Field(3, description="Number of similar tickets to retrieve", ge=1, le=20)
    similarity_threshold: Optional[float] = Field(0.3, description="Minimum similarity score (0-1)", ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Someone stole my credit card",
                "model_type": "ensemble",
                "return_details": True,
                "top_k": 5,
                "similarity_threshold": 0.4
            }
        }


class ClassifyResponse(BaseModel):
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


class BatchFastRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of ticket texts")
    model_type: Optional[str] = "ensemble"
    return_details: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.65
    use_llm_fallback: Optional[bool] = True


class BatchFastResponse(BaseModel):
    results: List[ClassifyResponse]


class BatchClassifyRequest(BaseModel):
    tickets: List[ClassifyRequest] = Field(..., description="List of tickets to classify")


class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    models_available: List[str] = Field(..., description="List of loaded models")