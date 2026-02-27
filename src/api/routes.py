"""
API routes for ticket classification.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
from src.api.schemas import (
    ClassifyTicketRequest,
    ClassifyTicketResponse,
    HealthResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["tickets"])


@router.post(
    "/classify",
    response_model=ClassifyTicketResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a support ticket",
)
async def classify_ticket(request: ClassifyTicketRequest) -> ClassifyTicketResponse:
    """
    Classify a support ticket.

    Args:
        request: Ticket classification request

    Returns:
        Ticket classification response
    """
    try:
        logger.info(f"Classifying ticket: {request.ticket_id}")

        # TODO: Implement actual classification logic
        # For now, return placeholder response
        classification = "Bug Report"
        confidence = 0.85

        logger.info(f"Ticket {request.ticket_id} classified as {classification}")

        return ClassifyTicketResponse(
            ticket_id=request.ticket_id,
            classification=classification,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Error classifying ticket: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing ticket",
        )


@router.post(
    "/classify/batch",
    response_model=List[ClassifyTicketResponse],
    status_code=status.HTTP_200_OK,
    summary="Classify multiple support tickets",
)
async def classify_tickets_batch(
    requests: List[ClassifyTicketRequest],
) -> List[ClassifyTicketResponse]:
    """
    Classify multiple support tickets.

    Args:
        requests: List of ticket classification requests

    Returns:
        List of ticket classification responses
    """
    try:
        logger.info(f"Classifying {len(requests)} tickets")

        responses = []
        for request in requests:
            response = await classify_ticket(request)
            responses.append(response)

        return responses

    except Exception as e:
        logger.error(f"Error classifying tickets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing tickets",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status response
    """
    return HealthResponse(status="healthy", message="Service is running")
