"""
API routes for ticket classification – uses trained models with LLM fallback and RAG.
Includes monitoring, robust error handling, explicit type conversion, and fast batch endpoint.
"""
import time
import uuid
import json
import traceback
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from typing import List, Optional
from pathlib import Path
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import (
    ClassifyRequest,
    RagExplainRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
    BatchFastRequest,
    BatchFastResponse,
)
from src.api.classifier import ProductionTicketClassifier, EnsembleTicketClassifier
from src.rag.retriever import TicketRetriever
from src.rag.llm_fallback import LLMFallback
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["tickets"])

# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------
REQUEST_COUNT = Counter(
    'ticket_classification_requests_total',
    'Total classification requests',
    ['model_type', 'category', 'fraud']
)
REQUEST_LATENCY = Histogram(
    'ticket_classification_latency_seconds',
    'Request latency',
    ['model_type']
)
FRAUD_COUNT = Counter(
    'fraud_detected_total',
    'Total fraud detections'
)
RAG_REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG /explain requests',
    ['model_type', 'category']
)
RAG_LATENCY = Histogram(
    'rag_latency_seconds',
    'RAG endpoint latency',
    ['model_type']
)

# ------------------------------------------------------------------
# Global model instances
# ------------------------------------------------------------------
baseline_model = None
transformer_model = None
ensemble_model = None
retriever = None
llm_explainer = None


def init_models():
    """Initialize all models once at startup."""
    global baseline_model, transformer_model, ensemble_model, retriever, llm_explainer
    project_root = Path(__file__).resolve().parents[2]
    baseline_path = project_root / "models" / "saved" / "baseline"
    transformer_path = project_root / "models" / "saved" / "transformer"

    logger.info(f"Looking for baseline model at: {baseline_path}")
    logger.info(f"Looking for transformer model at: {transformer_path}")

    try:
        baseline_model = ProductionTicketClassifier(baseline_path, model_type="baseline")
        logger.info("Baseline model loaded")
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")

    try:
        transformer_model = ProductionTicketClassifier(transformer_path, model_type="transformer")
        logger.info("Transformer model loaded")
    except Exception as e:
        logger.error(f"Failed to load transformer model: {e}")

    if baseline_model and transformer_model:
        try:
            ensemble_model = EnsembleTicketClassifier(baseline_path, transformer_path)
            logger.info("Ensemble model created")
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}")
    else:
        if baseline_model or transformer_model:
            ensemble_model = EnsembleTicketClassifier(baseline_path, transformer_path)
            logger.info("Ensemble model created (fallback mode)")

    # Initialize retriever (Chroma)
    try:
        retriever = TicketRetriever()
        logger.info("Retriever initialized")
    except Exception as e:
        logger.warning(f"Could not initialize retriever: {e}")

    # Initialize LLM explainer
    try:
        llm_explainer = LLMFallback()
        if not llm_explainer.is_available():
            logger.warning("LLM explainer not available (missing API key)")
        else:
            logger.info("LLM explainer initialized")
    except Exception as e:
        logger.warning(f"Could not initialize LLM explainer: {e}")


def get_classifier(model_type: str):
    """Return the requested classifier instance."""
    if model_type == "baseline":
        if baseline_model is None:
            raise ValueError("Baseline model not available")
        return baseline_model
    elif model_type == "transformer":
        if transformer_model is None:
            raise ValueError("Transformer model not available")
        return transformer_model
    elif model_type == "ensemble":
        if ensemble_model is None:
            if baseline_model:
                logger.warning("Ensemble not available, using baseline")
                return baseline_model
            raise ValueError("Ensemble model not available")
        return ensemble_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def log_request(request_id, model_type, category, confidence, latency, needs_review, fraud, llm_fallback_used):
    """Structured logging for monitoring."""
    log_entry = {
        "event": "classification",
        "request_id": request_id,
        "model_type": model_type,
        "category": category,
        "confidence": confidence,
        "latency_ms": round(latency * 1000, 2),
        "needs_review": needs_review,
        "fraud": fraud,
        "llm_fallback_used": llm_fallback_used,
    }
    logger.info(json.dumps(log_entry))


# ------------------------------------------------------------------
# Classification endpoints
# ------------------------------------------------------------------
@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single ticket",
)
async def classify_ticket(request: ClassifyRequest) -> ClassifyResponse:
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    try:
        classifier = get_classifier(request.model_type)
        
        if request.return_details:
            result = classifier.predict(
                request.text,
                return_details=True,
                allow_llm_fallback=request.use_llm_fallback,
                confidence_threshold=request.confidence_threshold
            )
            if isinstance(result, tuple) and len(result) == 4:
                category, confidence, needs_review, model_used = result
            else:
                category = str(result)
                confidence = 0.0
                needs_review = True
                model_used = "error"
        else:
            category = classifier.predict(
                request.text,
                return_details=False,
                allow_llm_fallback=request.use_llm_fallback,
                confidence_threshold=request.confidence_threshold
            )
            confidence = None
            needs_review = None
            model_used = None

        category = str(category)
        if confidence is not None:
            confidence = float(confidence)
        if needs_review is not None:
            needs_review = bool(needs_review)
        if model_used is not None:
            model_used = str(model_used)

        latency = time.time() - start_time
        fraud = (category == "Fraud")
        if fraud:
            FRAUD_COUNT.inc()
        REQUEST_COUNT.labels(model_type=request.model_type, category=category, fraud=str(fraud)).inc()
        REQUEST_LATENCY.labels(model_type=request.model_type).observe(latency)
        log_request(request_id, request.model_type, category, confidence, latency, needs_review, fraud, (model_used == "llm"))

        if request.return_details:
            return ClassifyResponse(
                category=category,
                confidence=confidence,
                needs_review=needs_review,
                model_used=model_used
            )
        else:
            return ClassifyResponse(category=category)
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/classify/batch_fast",
    response_model=BatchFastResponse,
    status_code=status.HTTP_200_OK,
    summary="Fast batch classification (single API call)",
)
async def classify_batch_fast(request: BatchFastRequest) -> BatchFastResponse:
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    try:
        classifier = get_classifier(request.model_type)
        results = classifier.predict_batch(
            request.texts,
            return_details=request.return_details,
            allow_llm_fallback=request.use_llm_fallback,
            confidence_threshold=request.confidence_threshold
        )
        responses = []
        for res in results:
            if request.return_details:
                cat, conf, review, used = res
                responses.append(ClassifyResponse(category=cat, confidence=conf, needs_review=review, model_used=used))
            else:
                responses.append(ClassifyResponse(category=res))
        latency = time.time() - start_time
        logger.info(f"Batch fast {request_id}: {len(request.texts)} texts in {latency:.2f}s")
        return BatchFastResponse(results=responses)
    except Exception as e:
        logger.error(f"Batch fast {request_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/classify/batch",
    response_model=BatchClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple tickets (legacy)",
)
async def classify_tickets_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    results = []
    for ticket_req in request.tickets:
        start = time.time()
        rid = str(uuid.uuid4())[:8]
        try:
            classifier = get_classifier(ticket_req.model_type)
            if ticket_req.return_details:
                category, confidence, needs_review, model_used = classifier.predict(
                    ticket_req.text,
                    return_details=True,
                    allow_llm_fallback=ticket_req.use_llm_fallback,
                    confidence_threshold=ticket_req.confidence_threshold
                )
                category = str(category)
                confidence = float(confidence) if confidence is not None else None
                needs_review = bool(needs_review) if needs_review is not None else None
                model_used = str(model_used) if model_used is not None else None
            else:
                category = classifier.predict(
                    ticket_req.text,
                    return_details=False,
                    allow_llm_fallback=ticket_req.use_llm_fallback,
                    confidence_threshold=ticket_req.confidence_threshold
                )
                confidence = None
                needs_review = None
                model_used = None
                category = str(category)

            latency = time.time() - start
            fraud = (category == "Fraud")
            if fraud:
                FRAUD_COUNT.inc()
            REQUEST_COUNT.labels(model_type=ticket_req.model_type, category=category, fraud=str(fraud)).inc()
            REQUEST_LATENCY.labels(model_type=ticket_req.model_type).observe(latency)
            log_request(rid, ticket_req.model_type, category, confidence, latency, needs_review, fraud, (model_used == "llm"))

            if ticket_req.return_details:
                results.append(ClassifyResponse(
                    category=category,
                    confidence=confidence,
                    needs_review=needs_review,
                    model_used=model_used,
                ))
            else:
                results.append(ClassifyResponse(category=category))
        except Exception as e:
            logger.error(f"Batch item {rid} failed: {e}")
            results.append(ClassifyResponse(category="Error", confidence=0.0, needs_review=True, model_used="error"))
    return BatchClassifyResponse(results=results)


# ------------------------------------------------------------------
# RAG endpoint
# ------------------------------------------------------------------
@router.post(
    "/rag/explain",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Explain classification with similar tickets",
)
async def rag_explain(request: RagExplainRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    try:
        classifier = get_classifier(request.model_type)
        category, confidence, needs_review, model_used = classifier.predict(
            request.text,
            return_details=True,
            allow_llm_fallback=request.use_llm_fallback,
            confidence_threshold=request.confidence_threshold
        )

        similar_tickets = []
        if retriever:
            try:
                similar_tickets = retriever.retrieve(
                    request.text,
                    top_k=request.top_k,
                    score_threshold=request.similarity_threshold
                )
            except Exception as e:
                logger.warning(f"Retrieval failed for {request_id}: {e}")

        explanation = None
        if llm_explainer and llm_explainer.is_available():
            try:
                explanation = llm_explainer.explain_prediction(request.text, category, similar_tickets)
            except Exception as e:
                logger.warning(f"Explanation failed for {request_id}: {e}")
                explanation = "LLM explanation temporarily unavailable."

        similar_formatted = []
        for t in similar_tickets:
            similar_formatted.append({
                "score": float(t.get("score", 0)),
                "category": str(t.get("metadata", {}).get("category", "unknown")),
                "text_preview": str(t.get("metadata", {}).get("clean_text", "")[:200]),
            })

        latency = time.time() - start_time
        RAG_REQUEST_COUNT.labels(model_type=request.model_type, category=category).inc()
        RAG_LATENCY.labels(model_type=request.model_type).observe(latency)
        logger.info(json.dumps({
            "event": "rag_explain",
            "request_id": request_id,
            "model_type": request.model_type,
            "category": category,
            "confidence": confidence,
            "latency_ms": round(latency * 1000, 2),
        }))

        response_data = {
            "classification": {
                "category": str(category),
                "confidence": float(confidence),
                "needs_review": bool(needs_review),
                "model_used": str(model_used),
            },
            "similar_tickets": similar_formatted,
            "explanation": str(explanation) if explanation else "No explanation available.",
        }
        return response_data
    except Exception as e:
        logger.error(f"RAG explain error {request_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Metrics and health
# ------------------------------------------------------------------
@router.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    models_available = []
    if baseline_model:
        models_available.append("baseline")
    if transformer_model:
        models_available.append("transformer")
    if ensemble_model:
        models_available.append("ensemble")
    return HealthResponse(status="healthy", models_available=models_available)