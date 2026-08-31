"""
Razorpay Webhook Verification and Event Normalization Module.

Provides cryptographic HMAC-SHA256 signature verification and defensive event normalization
for Razorpay payment link and payment webhook events.
"""

import os
import hmac
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

from backend.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_WEBHOOK_EVENTS = {
    "payment_link.paid",
    "payment.failed",
    "payment_link.expired",
}


@dataclass
class NormalizedWebhookEvent:
    """Standardized internal representation of a Razorpay webhook event."""
    event_type: str
    payment_link_id: Optional[str] = None
    payment_id: Optional[str] = None
    reference_id: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "INR"
    status: Optional[str] = None
    error_code: Optional[str] = None
    error_description: Optional[str] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


def get_webhook_secret() -> str:
    """Retrieves the configured Razorpay webhook secret from settings or environment."""
    return os.getenv("RAZORPAY_WEBHOOK_SECRET") or settings.RAZORPAY_WEBHOOK_SECRET or ""


def verify_razorpay_webhook_signature(
    raw_body: Union[bytes, str],
    signature: str,
    secret: Optional[str] = None
) -> bool:
    """
    Cryptographically verifies the Razorpay webhook signature header using HMAC-SHA256.

    Uses constant-time string comparison to prevent timing attacks.
    Never trusts incoming webhook payloads prior to signature verification.
    """
    webhook_secret = secret if secret is not None else get_webhook_secret()

    if not signature or not webhook_secret:
        return False

    if isinstance(raw_body, str):
        body_bytes = raw_body.encode("utf-8")
    elif isinstance(raw_body, (bytes, bytearray)):
        body_bytes = bytes(raw_body)
    else:
        return False

    secret_bytes = webhook_secret.encode("utf-8") if isinstance(webhook_secret, str) else webhook_secret

    computed_digest = hmac.new(secret_bytes, body_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed_digest, signature.strip())


def normalize_razorpay_webhook(payload: Dict[str, Any]) -> NormalizedWebhookEvent:
    """
    Defensively extracts correlation identifiers, payment status, and metadata
    from a Razorpay webhook event payload.
    """
    event_type = payload.get("event", "unknown")
    event_payload = payload.get("payload", {})

    payment_link_entity = event_payload.get("payment_link", {}).get("entity", {})
    payment_entity = event_payload.get("payment", {}).get("entity", {})

    # Extract payment link identifier (Primary correlation ID)
    payment_link_id = (
        payment_link_entity.get("id")
        or payment_entity.get("payment_link_id")
        or None
    )

    # Extract internal reference / transaction ID (Secondary correlation ID)
    reference_id = (
        payment_link_entity.get("reference_id")
        or payment_entity.get("notes", {}).get("reference_id")
        or payment_entity.get("notes", {}).get("transaction_id")
        or None
    )

    # Extract Razorpay payment ID
    payment_id = payment_entity.get("id") or None

    # Amount: Razorpay amounts are in paise (convert to INR float)
    raw_amount = (
        payment_link_entity.get("amount_paid")
        or payment_link_entity.get("amount")
        or payment_entity.get("amount")
    )
    amount = float(raw_amount) / 100.0 if raw_amount is not None else None

    currency = payment_link_entity.get("currency") or payment_entity.get("currency") or "INR"
    status = payment_link_entity.get("status") or payment_entity.get("status") or None

    error_code = payment_entity.get("error_code")
    error_description = payment_entity.get("error_description")

    return NormalizedWebhookEvent(
        event_type=event_type,
        payment_link_id=payment_link_id,
        payment_id=payment_id,
        reference_id=reference_id,
        amount=amount,
        currency=currency,
        status=status,
        error_code=error_code,
        error_description=error_description,
        raw_payload=payload
    )
