"""
Razorpay Webhook API Endpoints for RecoverAI.

Handles secure asynchronous webhook delivery for Razorpay payment links and payment outcomes,
closing the loop between recovery action initiation and financial recovery confirmation.
"""

import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database.database import get_db
from database.repository import process_webhook_lifecycle_event
from payment.webhook import (
    verify_razorpay_webhook_signature,
    normalize_razorpay_webhook,
    get_webhook_secret,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/razorpay",
    status_code=status.HTTP_200_OK,
    summary="Process Razorpay Payment Lifecycle Webhooks",
    description="Validates cryptographic HMAC-SHA256 signature on raw request body, normalizes event, correlates with internal recovery claim, and updates transaction lifecycle."
)
async def razorpay_webhook(
    request: Request,
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Asynchronous webhook handler for Razorpay events (e.g. payment_link.paid, payment.failed, payment_link.expired).
    """
    # 1. Read raw request body bytes for strict cryptographic verification
    raw_body = await request.body()

    # 2. Extract X-Razorpay-Signature header
    signature = request.headers.get("X-Razorpay-Signature")
    if not signature:
        logger.warning("Webhook received without X-Razorpay-Signature header.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Missing X-Razorpay-Signature header.",
                "error_code": "MISSING_WEBHOOK_SIGNATURE"
            }
        )

    # 3. Cryptographically verify signature
    is_valid = verify_razorpay_webhook_signature(raw_body, signature)
    if not is_valid:
        logger.warning("Webhook rejected: Invalid signature.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Invalid webhook signature.",
                "error_code": "INVALID_WEBHOOK_SIGNATURE"
            }
        )

    # 4. Safely parse JSON body
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        logger.warning(f"Webhook rejected: Malformed JSON ({exc}).")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Malformed JSON payload in webhook body.",
                "error_code": "MALFORMED_WEBHOOK_PAYLOAD"
            }
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Webhook body must be a valid JSON object.",
                "error_code": "INVALID_PAYLOAD_STRUCTURE"
            }
        )

    # 5. Normalize event into standardized internal representation
    event = normalize_razorpay_webhook(payload)

    # 6. Correlate with database records and execute atomic state transitions
    result = process_webhook_lifecycle_event(db, event)
    outcome_status = result.get("status")

    if outcome_status == "unmatched":
        logger.warning(f"Webhook correlation failed: {result.get('message')}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "status": "error",
                "message": result.get("message"),
                "error_code": "CLAIM_NOT_FOUND",
                "event_type": event.event_type
            }
        )

    elif outcome_status == "already_processed":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "already_processed",
                "message": result.get("message"),
                "transaction_id": result.get("transaction_id"),
                "claim_status": result.get("claim_status")
            }
        )

    elif outcome_status == "ignored":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ignored",
                "message": result.get("message"),
                "event_type": result.get("event_type")
            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result
    )
