from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.api.classifier import ProductionTicketClassifier
from pathlib import Path

app = FastAPI(title="Ticket Classifier API")

# Load model once at startup
classifier = ProductionTicketClassifier("models/saved/baseline")

class TicketRequest(BaseModel):
    text: str
    return_details: bool = False

class TicketResponse(BaseModel):
    category: str
    confidence: float = None
    method: str = None
    needs_review: bool = None

@app.post("/predict", response_model=TicketResponse)
async def predict(request: TicketRequest):
    try:
        if request.return_details:
            cat, conf, method, review = classifier.predict(request.text, return_details=True)
            return TicketResponse(category=cat, confidence=conf, method=method, needs_review=review)
        else:
            cat = classifier.predict(request.text)
            return TicketResponse(category=cat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}