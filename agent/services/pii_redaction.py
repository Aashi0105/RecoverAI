"""
RecoverAI Defensive PII Redaction Layer.

Provides pure, deterministic sanitization of transaction context before sending
payloads to external LLM providers (e.g., Groq, OpenAI).

Core Guarantees:
1. Pure function with copy.deepcopy: Input context dictionary is NEVER mutated.
2. Direct key redaction: Exact match for explicit sensitive personal identifiers.
3. Unstructured value scrubbing: Regex redaction of emails, phones, and card PANs.
4. Operational field preservation: Business IDs, amounts, probabilities, and scores
   remain 100% untouched.
"""

import copy
import re
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Regex Patterns for Unstructured Text Sanitization
# -----------------------------------------------------------------------------

# Standard RFC-compliant email address matching
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# Payment card PANs: 13-19 digits with optional spaces or hyphens
CARD_PAN_PATTERN = re.compile(r'\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}|\d{4}[ -]?\d{6}[ -]?\d{5}|\d{13,19})\b')

# Reasonable Indian and international mobile/phone numbers (10 digits, optional country/trunk prefix)
PHONE_PATTERN = re.compile(r'(?:\+?91[\s\-]?|0)?[6-9]\d{9}\b')

# -----------------------------------------------------------------------------
# Key Configuration Sets
# -----------------------------------------------------------------------------

# Exact normalized sensitive keys that directly expose customer personal identity
SENSITIVE_KEYS: Dict[str, str] = {
    "customer_email": "[REDACTED_EMAIL]",
    "email": "[REDACTED_EMAIL]",
    "customer_name": "[REDACTED_NAME]",
    "full_name": "[REDACTED_NAME]",
    "phone_number": "[REDACTED_PHONE]",
    "mobile_number": "[REDACTED_PHONE]",
    "phone": "[REDACTED_PHONE]",
    "mobile": "[REDACTED_PHONE]",
    "address": "[REDACTED_ADDRESS]"
}

# Critical operational and business telemetry fields that must NEVER be redacted or replaced
PROTECTED_OPERATIONAL_KEYS = {
    "transaction_id",
    "customer_id",
    "merchant_id",
    "amount",
    "currency",
    "payment_method",
    "payment_network",
    "payment_channel",
    "failure_category",
    "previous_failures_24h",
    "previous_failures_7d",
    "consecutive_failure_streak",
    "recovery_attempt_count",
    "recovery_probability",
    "expected_recovery_value",
    "ip_risk_score",
    "velocity_score",
    "risk_level"
}


def redact_text_pii(text: str) -> str:
    """
    Sanitizes embedded personal data (card numbers, emails, phone numbers)
    from unstructured string descriptions.
    """
    if not isinstance(text, str):
        return text

    # Process cards first to prevent overlap with phone numbers
    sanitized = CARD_PAN_PATTERN.sub("[REDACTED_CARD]", text)
    sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
    sanitized = PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)
    return sanitized


def _redact_value(val: Any) -> Any:
    """Recursively processes dictionary values, lists, and strings."""
    if isinstance(val, dict):
        return _redact_dict(val)
    elif isinstance(val, list):
        return [_redact_value(item) for item in val]
    elif isinstance(val, str):
        return redact_text_pii(val)
    return val


def _redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively redacts dictionary keys and values.
    Preserves protected operational keys while scrubbing direct PII keys and unstructured text.
    """
    redacted: Dict[str, Any] = {}
    for key, value in d.items():
        norm_key = str(key).strip().lower()

        # 1. Exact sensitive direct PII key match
        if norm_key in SENSITIVE_KEYS:
            redacted[key] = SENSITIVE_KEYS[norm_key]

        # 2. Protected operational identifier: preserve value untouched
        elif norm_key in PROTECTED_OPERATIONAL_KEYS:
            redacted[key] = value

        # 3. All other keys (e.g. failure_reason, diagnosis, notes): recursively scrub text/structures
        else:
            redacted[key] = _redact_value(value)

    return redacted


def redact_pii_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces a sanitized, deep-copied dictionary safe for external LLM prompt inclusion.
    Guarantees the input context dictionary remains completely unmodified.
    """
    if not context or not isinstance(context, dict):
        return {}

    # Deep copy guarantees zero mutation of caller's original data
    safe_copy = copy.deepcopy(context)
    return _redact_dict(safe_copy)
