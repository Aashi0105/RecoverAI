"""
Razorpay TEST MODE Client Wrapper for RecoverAI.

Handles safe creation of Razorpay Test Mode Payment Links using environment variables.
Never hardcodes credentials or processes live real-world currency.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Load credentials strictly from environment variables
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_ENABLED = os.getenv("RAZORPAY_ENABLED", "false").lower() == "true"


def is_razorpay_configured() -> bool:
    """
    Checks whether valid Razorpay TEST MODE credentials exist in environment.
    """
    return bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET and RAZORPAY_KEY_ID.startswith("rzp_test_"))


def get_razorpay_client():
    """
    Returns an initialized Razorpay Client instance if configured.
    """
    if not is_razorpay_configured():
        raise ValueError("Razorpay TEST MODE credentials not configured or invalid key_id format (must start with rzp_test_).")
    
    import razorpay
    return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


def create_razorpay_test_payment_link(
    amount: float,
    customer_info: Dict[str, str],
    description: str = "RecoverAI Payment Recovery",
    reference_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a real Razorpay TEST MODE Payment Link via API.
    Amount converted to paise (INR * 100).
    """
    client = get_razorpay_client()
    amount_in_paise = int(round(amount * 100))

    payload = {
        "amount": amount_in_paise,
        "currency": "INR",
        "accept_partial": False,
        "description": description,
        "customer": {
            "name": customer_info.get("name", "Valued Customer"),
            "email": customer_info.get("email", "customer@example.com"),
            "contact": customer_info.get("contact", "+919999999999")
        },
        "notify": {
            "sms": True,
            "email": True
        },
        "reminder_enable": True,
        "notes": {
            "agent": "RecoverAI",
            "environment": "test_mode"
        }
    }

    if reference_id:
        payload["reference_id"] = str(reference_id)

    try:
        response = client.payment_link.create(payload)
        return {
            "success": True,
            "payment_link_id": response.get("id"),
            "short_url": response.get("short_url"),
            "status": response.get("status"),
            "amount_paid": response.get("amount_paid", 0) / 100.0,
            "raw_response": response
        }
    except Exception as e:
        logger.error(f"Razorpay API Error: {e}")
        return {
            "success": False,
            "error_message": str(e),
            "payment_link_id": None,
            "short_url": None
        }
