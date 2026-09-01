"""
Unit & Integration Test Suite for Phase 3A: Real Agent Intelligence with Safe Heuristic Fallback.

Tests:
1. LLM disabled (LLM_ENABLED=false) -> Fallback to heuristics.
2. Missing or placeholder API key -> Fallback to heuristics without crash.
3. Successful LLM diagnosis -> diagnosis_source == 'llm'.
4. Successful LLM recommendation -> recommendation_source == 'llm'.
5. Invalid LLM JSON -> Safe fallback to heuristics.
6. Unsupported LLM action -> Rejected and safely falls back to heuristics.
7. LLM provider network exception / timeout -> Graceful fallback without agent failure.
8. CRITICAL FINTECH INVARIANT: Policy guard overrides LLM recommendation.
"""

from unittest.mock import patch
import pytest

from agent.graph import run_agent
from agent.nodes.policy import policy_guard
from agent.nodes.diagnosis import diagnose_failure
from agent.nodes.recommendation import recommend_action
from agent.demo_data import build_test_transaction
from agent.services.llm_service import is_llm_available, DiagnosisOutput, RecommendationOutput


def get_base_test_transaction():
    """Generates a standard test transaction dictionary."""
    return build_test_transaction(
        transaction_id="txn_llm_test_001",
        customer_id="cust_llm_001",
        merchant_id="merch_01",
        amount=2500.0,
        currency="INR",
        failure_reason="network_timeout",
        failure_category="transient",
        customer_historical_success_rate=0.92,
        previous_failures_24h=0,
        recovery_attempt_count=0,
        ip_risk_score=0.05,
        velocity_score=0.10
    )


# -----------------------------------------------------------------------------
# 1. Availability and Configuration Checks
# -----------------------------------------------------------------------------

def test_llm_disabled_defaults_to_heuristic(monkeypatch):
    """Verifies that when LLM_ENABLED is false, is_llm_available returns False and agent runs heuristics."""
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "dummy_key_123")

    assert is_llm_available() is False

    txn = get_base_test_transaction()
    res = run_agent(txn)

    assert res["diagnosis_source"] == "heuristic"
    assert res["recommendation_source"] == "heuristic"
    assert res["policy_decision"] == "ACT"
    assert res["agent_status"] in ["APPROVED", "COMPLETED"]


def test_missing_api_key_defaults_to_heuristic(monkeypatch):
    """Verifies that missing or placeholder API key does not crash and defaults to heuristics."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "")

    assert is_llm_available() is False

    txn = get_base_test_transaction()
    res = run_agent(txn)

    assert res["diagnosis_source"] == "heuristic"
    assert res["recommendation_source"] == "heuristic"


def test_placeholder_api_key_defaults_to_heuristic(monkeypatch):
    """Verifies that default placeholder key 'your_llm_api_key_here' is treated as unconfigured."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "your_llm_api_key_here")

    assert is_llm_available() is False


# -----------------------------------------------------------------------------
# 2. Successful LLM Structured Responses
# -----------------------------------------------------------------------------

def test_successful_llm_diagnosis_integration(monkeypatch):
    """Verifies that valid structured output from LLM is stored and diagnosis_source == 'llm'."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    mock_llm_diag = DiagnosisOutput(
        failure_category="transient",
        diagnosis="Bank gateway experiencing intermittent network drop during 3D Secure verification.",
        severity="LOW",
        customer_action_required=False,
        key_factors=["intermittent gateway timeout", "low 24h failure count"],
        confidence=0.91
    )

    with patch("agent.nodes.diagnosis.generate_llm_diagnosis", return_value=mock_llm_diag):
        state = get_base_test_transaction()
        updated_state = diagnose_failure(state)

        assert updated_state["diagnosis_source"] == "llm"
        diag = updated_state["failure_diagnosis"]
        assert diag["diagnosis"] == "Bank gateway experiencing intermittent network drop during 3D Secure verification."
        assert diag["severity"] == "LOW"
        assert "intermittent gateway timeout" in diag["key_factors"]
        assert diag["confidence"] == 0.91


def test_successful_llm_recommendation_integration(monkeypatch):
    """Verifies that valid structured recommendation from LLM is applied and recommendation_source == 'llm'."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    mock_llm_rec = RecommendationOutput(
        recommended_action="payment_link",
        decision_rationale="Alternative UPI/Card payment link recommended to bypass current card gateway latency.",
        key_factors=["customer card gateway latency", "high probability customer (92%)"],
        confidence=0.88,
        expected_benefit="Allows subscriber to quickly pay via alternative UPI app without waiting."
    )

    with patch("agent.nodes.recommendation.generate_llm_recommendation", return_value=mock_llm_rec):
        state = get_base_test_transaction()
        updated_state = recommend_action(state)

        assert updated_state["recommendation_source"] == "llm"
        assert updated_state["recommended_action"] == "payment_link"
        assert "Alternative UPI/Card payment link" in updated_state["recommendation_reason"]
        assert updated_state["recommendation_confidence"] == 0.88
        assert "customer card gateway latency" in updated_state["recommendation_factors"]


# -----------------------------------------------------------------------------
# 3. Defensive Validation & Fallback Handling
# -----------------------------------------------------------------------------

def test_invalid_llm_json_falls_back_safely(monkeypatch):
    """Verifies that malformed JSON or schema rejection falls back to heuristics without crashing."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    # Simulate raw API returning invalid dictionary or None
    with patch("agent.services.llm_service.call_llm_json", return_value=None):
        state = get_base_test_transaction()
        updated_diag = diagnose_failure(state)
        assert updated_diag["diagnosis_source"] == "heuristic"

        updated_rec = recommend_action(state)
        assert updated_rec["recommendation_source"] == "heuristic"


def test_unsupported_llm_action_rejected(monkeypatch):
    """Verifies that an unsupported action (e.g. 'charge_customer_again') is rejected and falls back."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    # Raw payload returns unauthorized action outside ALLOWED_ACTIONS
    invalid_raw_payload = {
        "recommended_action": "charge_customer_again",
        "decision_rationale": "Force charge customer without permission.",
        "key_factors": ["aggressive billing"],
        "confidence": 0.99
    }

    with patch("agent.services.llm_service.call_llm_json", return_value=invalid_raw_payload):
        state = get_base_test_transaction()
        updated_state = recommend_action(state)

        # Must fall back to safe allowed heuristic action!
        assert updated_state["recommendation_source"] == "heuristic"
        assert updated_state["recommended_action"] in {"retry", "payment_link", "reminder", "escalate", "no_action"}
        assert updated_state["recommended_action"] != "charge_customer_again"


def test_llm_provider_network_exception_handled_gracefully(monkeypatch):
    """Verifies that network exceptions or timeouts during LLM calls do not fail the agent."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "valid_test_key_xyz")

    with patch("agent.services.llm_service.call_llm_json", side_effect=Exception("Connection timed out after 6.0s")):
        txn = get_base_test_transaction()
        res = run_agent(txn)

        assert res["diagnosis_source"] == "heuristic"
        assert res["recommendation_source"] == "heuristic"
        assert res["policy_decision"] == "ACT"


# -----------------------------------------------------------------------------
# 4. CRITICAL FINTECH SAFETY INVARIANT: Policy Guard Overrides LLM
# -----------------------------------------------------------------------------

def test_policy_guard_strictly_overrides_llm_recommendation(monkeypatch):
    """
    CRITICAL FINTECH SAFETY INVARIANT:
    Even if the LLM recommends 'retry' with 99% confidence for a suspicious transaction,
    the deterministic Policy Guard MUST REFUSE/BLOCK the transaction.
    'LLM recommends. Deterministic policy controls.'
    """
    state = build_test_transaction(
        transaction_id="txn_adversarial_llm_001",
        amount=3000.0,
        failure_category="transient",
        failure_reason="network_timeout",
        ip_risk_score=0.85,  # High risk!
        previous_failures_24h=3,
        consecutive_failure_streak=4  # Exceeds streak limit!
    )

    state["recovery_probability"] = 0.95
    state["recommended_action"] = "retry"
    state["recommendation_source"] = "llm"
    state["recommendation_confidence"] = 0.99

    # Execute Policy Guard
    guarded_state = policy_guard(state)

    # Invariant checks
    assert guarded_state["policy_decision"] == "REFUSE"
    assert guarded_state["agent_status"] == "BLOCKED"
    assert guarded_state["selected_action"] is None
    assert any("streak" in v.lower() or "risk" in v.lower() for v in guarded_state["policy_violations"])


def test_high_value_transaction_escalates_regardless_of_llm_recommendation():
    """
    CRITICAL FINTECH SAFETY INVARIANT:
    Even if the LLM recommends automated 'payment_link', transactions exceeding ₹8,500
    MUST escalate to AWAITING_APPROVAL for merchant review.
    """
    state = build_test_transaction(
        transaction_id="txn_high_value_llm_002",
        amount=15000.0,  # Exceeds ₹8,500 threshold
        failure_category="transient",
        failure_reason="network_timeout",
        ip_risk_score=0.05
    )

    state["recovery_probability"] = 0.85
    state["recommended_action"] = "payment_link"
    state["recommendation_source"] = "llm"

    guarded_state = policy_guard(state)

    assert guarded_state["policy_decision"] == "ESCALATE"
    assert guarded_state["agent_status"] == "AWAITING_APPROVAL"
    assert guarded_state["selected_action"] is None
