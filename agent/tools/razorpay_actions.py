"""
Razorpay Test Mode API Actions & Client Wrapper.

Provides real integration with official Razorpay Python SDK for Test Mode actions:
- Payment Links API (customer action required / card issues)
- Orders API (payment retries)
- Provider status queries

Converts amounts correctly from INR to paise (amount * 100).
Never logs or exposes secret keys.
"""

import os
from typing import Dict, Any, Optional
import razorpay
from backend.config import settings


def get_razorpay_client() -> razorpay.Client:
    """
    Initializes and returns official Razorpay SDK client using configured credentials.
    Raises ValueError if credentials are not configured.
    """
    key_id = getattr(settings, "RAZORPAY_KEY_ID", None) or os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = getattr(settings, "RAZORPAY_KEY_SECRET", None) or os.environ.get("RAZORPAY_KEY_SECRET", "")

    if not key_id or not key_secret:
        raise ValueError("Razorpay Test Mode credentials (KEY_ID / KEY_SECRET) are missing or unconfigured.")

    return razorpay.Client(auth=(key_id, key_secret))


def generate_test_contact(customer_id: str) -> str:
    """Generates a valid, non-repeating 10-digit Indian mobile contact number for Razorpay Test Mode."""
    base_digits = "9876543210"
    cust_digits = "".join(filter(str.isdigit, customer_id))
    if len(cust_digits) >= 10:
        num_str = cust_digits[-10:]
        if len(set(num_str)) > 1:
            return f"+91{num_str}"
    return f"+91{base_digits}"


def razorpay_create_payment_link(
    transaction_id: str,
    amount: float,
    customer_id: str,
    description: str = ""
) -> Dict[str, Any]:
    """
    Creates a real Razorpay Test Mode Payment Link via the official SDK.
    Amount in INR is converted to paise (1 INR = 100 paise).
    """
    try:
        client = get_razorpay_client()
        amount_in_paise = int(round(amount * 100))
        contact_num = generate_test_contact(customer_id)

        payload = {
            "amount": amount_in_paise,
            "currency": "INR",
            "accept_partial": False,
            "description": description or f"RecoverAI Payment Recovery for Txn {transaction_id}",
            "customer": {
                "name": f"Customer {customer_id}",
                "contact": contact_num,
                "email": f"{customer_id}@example.com"
            },
            "notify": {"sms": False, "email": False},
            "reminder_enable": False,
            "notes": {
                "transaction_id": transaction_id,
                "system": "RecoverAI"
            }
        }

        res = client.payment_link.create(payload)
        link_id = res.get("id", "")
        short_url = res.get("short_url", "")
        raw_status = res.get("status", "created")

        return {
            "status": "executed",
            "action": "payment_link",
            "reference_id": link_id,
            "provider": "razorpay",
            "payment_link_id": link_id,
            "payment_link_url": short_url,
            "provider_status": raw_status,
            "message": f"Razorpay Test Mode payment link created: {short_url}"
        }

    except Exception as e:
        return {
            "status": "failed",
            "action": "payment_link",
            "reference_id": "NONE",
            "provider": "razorpay",
            "provider_status": "error",
            "error_detail": str(e),
            "message": f"Failed to create Razorpay Test Mode payment link: {str(e)}"
        }


def razorpay_create_recovery_order(
    transaction_id: str,
    amount: float,
    currency: str = "INR"
) -> Dict[str, Any]:
    """
    Creates a real Razorpay Test Mode Order via the official SDK for smart retries.
    Amount in INR is converted to paise (1 INR = 100 paise).
    """
    try:
        client = get_razorpay_client()
        amount_in_paise = int(round(amount * 100))

        # Truncate receipt name if necessary to comply with Razorpay receipt max length limit (40 chars)
        receipt_str = f"rcpt_{transaction_id}"[:35]

        payload = {
            "amount": amount_in_paise,
            "currency": currency,
            "receipt": receipt_str,
            "notes": {
                "transaction_id": transaction_id,
                "action": "retry",
                "system": "RecoverAI"
            }
        }

        res = client.order.create(payload)
        order_id = res.get("id", "")
        raw_status = res.get("status", "created")

        return {
            "status": "executed",
            "action": "retry",
            "reference_id": order_id,
            "provider": "razorpay",
            "order_id": order_id,
            "provider_status": raw_status,
            "message": f"Razorpay Test Mode recovery order created: {order_id}"
        }

    except Exception as e:
        return {
            "status": "failed",
            "action": "retry",
            "reference_id": "NONE",
            "provider": "razorpay",
            "provider_status": "error",
            "error_detail": str(e),
            "message": f"Failed to create Razorpay Test Mode recovery order: {str(e)}"
        }


def razorpay_fetch_payment_link_status(payment_link_id: str) -> Optional[Dict[str, Any]]:
    """Fetches real-time status of a Razorpay Payment Link."""
    try:
        client = get_razorpay_client()
        return client.payment_link.fetch(payment_link_id)
    except Exception:
        return None


def razorpay_fetch_order_status(order_id: str) -> Optional[Dict[str, Any]]:
    """Fetches real-time status of a Razorpay Order."""
    try:
        client = get_razorpay_client()
        return client.order.fetch(order_id)
    except Exception:
        return None
