"""
Node 4: Recovery Strategy Recommendation
Generates structured recovery action recommendation.
Supports real LLM structured strategy selection with safe deterministic heuristic fallback.
Allowed actions: 'retry', 'payment_link', 'reminder', 'escalate', 'no_action'.
"""

from typing import Dict, Any
from agent.state import AgentState
from agent.services.llm_service import (
    is_llm_available,
    generate_llm_recommendation,
    ALLOWED_ACTIONS
)


def heuristic_recommendation(state: AgentState) -> Dict[str, Any]:
    """
    Deterministic rule-based recovery recommendation.
    Provides 100% reliable fallback when LLM is unconfigured, disabled, or offline.
    """
    category = state.get("failure_category", "transient")
    prob = state.get("recovery_probability", 0.50)
    amount = state.get("amount", 0.0)
    customer_ctx = state.get("customer_context", {})
    prev_fails_24h = customer_ctx.get("previous_failures_24h", 0)
    ip_risk = customer_ctx.get("ip_risk_score", 0.0)

    if category == "risk_related" or ip_risk > 0.7:
        action = "escalate"
        reason = "Risk score is high or transaction flagged for fraud suspicion."
        confidence = 0.90
        benefit = "Prevents fraudulent chargebacks and protects merchant account."
        factors = ["high IP risk score", "fraud suspicion flag"]
    elif prob < 0.35:
        action = "no_action"
        reason = "ML recovery probability is very low (< 35%). Recovery attempt unlikely to succeed."
        confidence = 0.85
        benefit = "Saves customer notification fatigue and operational overhead."
        factors = [f"low ML probability ({prob:.2f})", "sub-threshold EV"]
    elif category == "transient" and prev_fails_24h < 2:
        action = "retry"
        reason = "Transient network issue with strong recovery probability. Smart retry recommended."
        confidence = 0.88
        benefit = "Immediate friction-free recovery without requiring customer effort."
        factors = ["transient network failure", f"prior failures 24h: {prev_fails_24h}"]
    elif category in ["customer_action_required", "payment_method_problem"]:
        if amount > 2000:
            action = "payment_link"
            reason = "Customer action/card issue detected. Sending alternative payment link."
            confidence = 0.82
            benefit = "Enables customer to update payment method or choose alternative UPI/Netbanking."
            factors = [f"amount ₹{amount:,.2f} > ₹2,000", "alternative payment instrument needed"]
        else:
            action = "reminder"
            reason = "Low-value transaction with customer action required. Sending friendly reminder."
            confidence = 0.78
            benefit = "Gentle nudge without high messaging cost."
            factors = [f"low amount ₹{amount:,.2f}", "customer action required"]
    elif amount > 25000:
        action = "escalate"
        reason = "High transaction value (> ₹25,000). Escalating for human merchant review."
        confidence = 0.95
        benefit = "Ensures high-value subscriber receives personalized touchpoint."
        factors = [f"high value transaction (₹{amount:,.2f})"]
    else:
        action = "retry"
        reason = "Standard retry recommendation based on moderate recovery probability."
        confidence = 0.75
        benefit = "Standard payment recovery attempt."
        factors = ["moderate ML probability", "standard recovery path"]

    return {
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "benefit": benefit,
        "factors": factors
    }


def recommend_action(state: AgentState) -> AgentState:
    """
    Produces structured action recommendation based on ML probability and context.
    Uses LLM structured output if enabled; otherwise falls back to heuristics.
    """
    # 1. Try real LLM recommendation if available
    if is_llm_available():
        context = {
            "transaction_id": state.get("transaction_id", "unknown"),
            "amount": state.get("amount", 0.0),
            "currency": state.get("currency", "INR"),
            "payment_method": state.get("payment_method", "unknown"),
            "failure_reason": state.get("failure_reason", "unknown"),
            "failure_category": state.get("failure_category", "unknown"),
            "recovery_probability": state.get("recovery_probability", 0.5),
            "expected_recovery_value": state.get("expected_recovery_value", 0.0),
            "risk_level": state.get("risk_level", "MEDIUM"),
            "failure_diagnosis": state.get("failure_diagnosis", {}),
            "previous_failures_24h": state.get("customer_context", {}).get("previous_failures_24h", state.get("previous_failures_24h", 0)),
            "consecutive_failure_streak": state.get("customer_context", {}).get("consecutive_failure_streak", state.get("consecutive_failure_streak", 0)),
            "ip_risk_score": state.get("customer_context", {}).get("ip_risk_score", state.get("ip_risk_score", 0.0))
        }

        try:
            llm_result = generate_llm_recommendation(context)
            if llm_result and llm_result.recommended_action in ALLOWED_ACTIONS:
                state["recommended_action"] = llm_result.recommended_action
                state["recommendation_reason"] = llm_result.decision_rationale
                state["recommendation_confidence"] = llm_result.confidence
                state["recommendation_factors"] = llm_result.key_factors
                state["recommendation_expected_benefit"] = llm_result.expected_benefit or ""
                state["recommendation_source"] = "llm"
                return state
        except Exception:
            pass  # Fall through safely to heuristic

    # 2. Deterministic Heuristic Fallback
    h_res = heuristic_recommendation(state)
    state["recommended_action"] = h_res["action"]
    state["recommendation_reason"] = h_res["reason"]
    state["recommendation_confidence"] = h_res["confidence"]
    state["recommendation_factors"] = h_res.get("factors", [])
    state["recommendation_expected_benefit"] = h_res.get("benefit", "")
    state["recommendation_source"] = "heuristic"
    return state
