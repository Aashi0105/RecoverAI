"""
Comprehensive Unit Tests for Feature 3: Defensive PII Redaction Layer.

Tests:
1. Input Immutability: Original dictionary is never modified.
2. Direct Sensitive Keys: Exact matches for sensitive PII keys are redacted.
3. Nested Data: Recursion across nested dictionaries, lists, and structures.
4. Email Pattern Redaction: Email addresses in free-form text are sanitized.
5. Phone Pattern Redaction: Indian and international phone numbers are sanitized.
6. Card Pattern Redaction: 13-19 digit card PANs with spaces/hyphens are sanitized.
7. Operational Field Preservation: Business telemetry and operational IDs remain untouched.
8. LLM Integration Test: Verifies that payloads passed to external LLM calls are sanitized.
"""

from unittest.mock import patch
import pytest

from agent.services.pii_redaction import redact_pii_from_context, redact_text_pii
from agent.services.llm_service import generate_llm_diagnosis, generate_llm_recommendation


def test_input_immutability():
    """Test 1: Verify the original context dictionary is never mutated."""
    original_context = {
        "transaction_id": "txn_immutable_001",
        "customer_id": "cust_999",
        "customer_email": "priya.nair@example.com",
        "phone_number": "+919876543210",
        "amount": 3450.0,
        "failure_reason": "Declined for priya.nair@example.com",
        "nested": {
            "email": "backup@example.com",
            "notes": ["Call +919876543210 immediately"]
        }
    }

    # Deep snapshot of original values for strict equality assertion
    snapshot_email = original_context["customer_email"]
    snapshot_phone = original_context["phone_number"]
    snapshot_reason = original_context["failure_reason"]
    snapshot_nested_email = original_context["nested"]["email"]
    snapshot_nested_note = original_context["nested"]["notes"][0]

    safe_context = redact_pii_from_context(original_context)

    # Safe context must be redacted
    assert safe_context["customer_email"] == "[REDACTED_EMAIL]"
    assert safe_context["phone_number"] == "[REDACTED_PHONE]"
    assert "[REDACTED_EMAIL]" in safe_context["failure_reason"]
    assert "priya.nair@example.com" not in safe_context["failure_reason"]
    assert safe_context["nested"]["email"] == "[REDACTED_EMAIL]"
    assert "[REDACTED_PHONE]" in safe_context["nested"]["notes"][0]

    # Original context MUST remain 100% untouched
    assert original_context["customer_email"] == snapshot_email
    assert original_context["phone_number"] == snapshot_phone
    assert original_context["failure_reason"] == snapshot_reason
    assert original_context["nested"]["email"] == snapshot_nested_email
    assert original_context["nested"]["notes"][0] == snapshot_nested_note


def test_direct_sensitive_keys():
    """Test 2: Verify direct sensitive keys are accurately redacted."""
    raw_context = {
        "customer_email": "user1@domain.com",
        "email": "user2@domain.com",
        "customer_name": "Aarav Sharma",
        "full_name": "Aarav Sharma",
        "phone_number": "+919123456780",
        "mobile_number": "9123456780",
        "phone": "+919123456780",
        "mobile": "9123456780",
        "address": "42 Marine Drive, Mumbai"
    }

    redacted = redact_pii_from_context(raw_context)

    assert redacted["customer_email"] == "[REDACTED_EMAIL]"
    assert redacted["email"] == "[REDACTED_EMAIL]"
    assert redacted["customer_name"] == "[REDACTED_NAME]"
    assert redacted["full_name"] == "[REDACTED_NAME]"
    assert redacted["phone_number"] == "[REDACTED_PHONE]"
    assert redacted["mobile_number"] == "[REDACTED_PHONE]"
    assert redacted["phone"] == "[REDACTED_PHONE]"
    assert redacted["mobile"] == "[REDACTED_PHONE]"
    assert redacted["address"] == "[REDACTED_ADDRESS]"


def test_nested_data():
    """Test 3: Verify recursion across nested dicts, lists, and diagnosis structures."""
    context = {
        "transaction_id": "txn_nest_123",
        "failure_diagnosis": {
            "diagnosis": "Card issue reported by customer ananya@startup.io",
            "key_factors": [
                "Transient timeout",
                "Contact cardholder at +919876543210"
            ]
        },
        "metadata_records": [
            {"email": "admin@merchant.com", "verified": True},
            {"phone": "9876543210", "attempts": 2}
        ]
    }

    safe = redact_pii_from_context(context)

    assert safe["transaction_id"] == "txn_nest_123"
    assert "ananya@startup.io" not in safe["failure_diagnosis"]["diagnosis"]
    assert "[REDACTED_EMAIL]" in safe["failure_diagnosis"]["diagnosis"]

    assert "+919876543210" not in safe["failure_diagnosis"]["key_factors"][1]
    assert "[REDACTED_PHONE]" in safe["failure_diagnosis"]["key_factors"][1]

    assert safe["metadata_records"][0]["email"] == "[REDACTED_EMAIL]"
    assert safe["metadata_records"][1]["phone"] == "[REDACTED_PHONE]"


def test_email_pattern_redaction():
    """Test 4: Verify email addresses in free-form error text are sanitized."""
    texts = [
        "Payment declined for rahul.verma@example.com by issuing bank",
        "Webhook alert: merchant_ops+alerts@sub.domain.co notified",
        "User contact: support-lead@company.org."
    ]

    for t in texts:
        sanitized = redact_text_pii(t)
        assert "@" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized


def test_phone_pattern_redaction():
    """Test 5: Verify Indian and international phone numbers are sanitized."""
    test_cases = [
        ("Customer call initiated to +919876543210", "[REDACTED_PHONE]"),
        ("SMS nudge sent to +91 9876543210", "[REDACTED_PHONE]"),
        ("Fallback contact: +91-9876543210", "[REDACTED_PHONE]"),
        ("Mobile number 9876543210 not answering", "[REDACTED_PHONE]"),
        ("Alternate number 09876543210 recorded", "[REDACTED_PHONE]")
    ]

    for raw, expected_marker in test_cases:
        sanitized = redact_text_pii(raw)
        assert "9876543210" not in sanitized
        assert expected_marker in sanitized


def test_card_pattern_redaction():
    """Test 6: Verify plausible 13-19 digit card PANs with spaces/hyphens are sanitized."""
    card_cases = [
        "Card 4111 1111 1111 1111 expired",
        "Card 4111-1111-1111-1111 expired",
        "Card 4111111111111111 expired",
        "Card 3782 822463 10005 flagged"  # 15-digit Amex style
    ]

    for c in card_cases:
        sanitized = redact_text_pii(c)
        assert "4111" not in sanitized or "[REDACTED_CARD]" in sanitized
        assert "[REDACTED_CARD]" in sanitized


def test_operational_field_preservation():
    """Test 7: Verify business telemetry and operational IDs remain 100% untouched."""
    operational_context = {
        "transaction_id": "txn_0000042",
        "customer_id": "cust_01879",
        "merchant_id": "merch_002",
        "amount": 4250.75,
        "currency": "INR",
        "payment_method": "upi",
        "failure_category": "transient",
        "previous_failures_24h": 1,
        "previous_failures_7d": 2,
        "consecutive_failure_streak": 3,
        "recovery_attempt_count": 1,
        "recovery_probability": 0.8845,
        "expected_recovery_value": 3759.79,
        "ip_risk_score": 0.12,
        "velocity_score": 0.18,
        "risk_level": "LOW"
    }

    safe = redact_pii_from_context(operational_context)

    # Operational fields must remain identical in key and value
    assert safe["transaction_id"] == "txn_0000042"
    assert safe["customer_id"] == "cust_01879"
    assert safe["merchant_id"] == "merch_002"
    assert safe["amount"] == 4250.75
    assert safe["currency"] == "INR"
    assert safe["payment_method"] == "upi"
    assert safe["failure_category"] == "transient"
    assert safe["previous_failures_24h"] == 1
    assert safe["previous_failures_7d"] == 2
    assert safe["consecutive_failure_streak"] == 3
    assert safe["recovery_attempt_count"] == 1
    assert safe["recovery_probability"] == 0.8845
    assert safe["expected_recovery_value"] == 3759.79
    assert safe["ip_risk_score"] == 0.12
    assert safe["velocity_score"] == 0.18
    assert safe["risk_level"] == "LOW"


def test_llm_integration_redaction(monkeypatch):
    """Test 8: Verify prompt payloads passed to external LLM calls are sanitized."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    context = {
        "transaction_id": "txn_llm_sec_001",
        "amount": 2500.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "Bank declined user dev@fintechcorp.in on phone +919876543210",
        "failure_category": "transient",
        "recovery_probability": 0.92,
        "customer_email": "dev@fintechcorp.in",
        "customer_name": "Dev Sharma"
    }

    captured_messages_diag = []
    captured_messages_rec = []

    def mock_call_diag(messages, timeout=6.0):
        captured_messages_diag.extend(messages)
        return {
            "failure_category": "transient",
            "diagnosis": "Network timeout with issuing bank.",
            "severity": "LOW",
            "customer_action_required": False,
            "key_factors": ["transient error"],
            "confidence": 0.90
        }

    def mock_call_rec(messages, timeout=6.0):
        captured_messages_rec.extend(messages)
        return {
            "recommended_action": "retry",
            "decision_rationale": "High recovery probability with transient failure.",
            "key_factors": ["transient error"],
            "confidence": 0.85
        }

    # 1. Test generate_llm_diagnosis
    with patch("agent.services.llm_service.call_llm_json", side_effect=mock_call_diag):
        diag_output = generate_llm_diagnosis(context)
        assert diag_output is not None

        user_content = captured_messages_diag[1]["content"]
        # Raw email and phone MUST NOT be present
        assert "dev@fintechcorp.in" not in user_content
        assert "+919876543210" not in user_content
        # Redaction markers MUST be present
        assert "[REDACTED_EMAIL]" in user_content
        assert "[REDACTED_PHONE]" in user_content
        # Operational business fields MUST be present
        assert "txn_llm_sec_001" in user_content
        assert "2500.0" in user_content

    # 2. Test generate_llm_recommendation
    with patch("agent.services.llm_service.call_llm_json", side_effect=mock_call_rec):
        rec_output = generate_llm_recommendation(context)
        assert rec_output is not None

        user_content = captured_messages_rec[1]["content"]
        assert "dev@fintechcorp.in" not in user_content
        assert "+919876543210" not in user_content
        assert "[REDACTED_EMAIL]" in user_content
        assert "[REDACTED_PHONE]" in user_content
        assert "txn_llm_sec_001" in user_content

    # 3. Confirm original context object remained 100% unchanged
    assert context["customer_email"] == "dev@fintechcorp.in"
    assert context["customer_name"] == "Dev Sharma"
    assert "dev@fintechcorp.in" in context["failure_reason"]
    assert "+919876543210" in context["failure_reason"]
