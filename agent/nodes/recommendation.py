"""
Node 4: Recovery Strategy Recommendation
Generates structured recovery action recommendation (LLM / Heuristic Fallback).
Allowed actions: 'retry', 'payment_link', 'reminder', 'escalate', 'no_action'.
"""

from agent.state import AgentState


def recommend_action(state: AgentState) -> AgentState:
    """
    Produces structured action recommendation based on ML probability and context.
    """
    category = state.get("failure_category", "transient")
    prob = state.get("recovery_probability", 0.50)
    amount = state.get("amount", 0.0)
    customer_ctx = state.get("customer_context", {})
    prev_fails_24h = customer_ctx.get("previous_failures_24h", 0)
    ip_risk = customer_ctx.get("ip_risk_score", 0.0)

    # Heuristic strategy recommendation (LLM fallback)
    if category == "risk_related" or ip_risk > 0.7:
        action = "escalate"
        reason = "Risk score is high or transaction flagged for fraud suspicion."
        confidence = 0.90
        benefit = "Prevents fraudulent chargebacks and protects merchant account."
    elif prob < 0.35:
        action = "no_action"
        reason = "ML recovery probability is very low (< 35%). Recovery attempt unlikely to succeed."
        confidence = 0.85
        benefit = "Saves customer notification fatigue and operational overhead."
    elif category == "transient" and prev_fails_24h < 2:
        action = "retry"
        reason = "Transient network issue with strong recovery probability. Smart retry recommended."
        confidence = 0.88
        benefit = "Immediate friction-free recovery without requiring customer effort."
    elif category in ["customer_action_required", "payment_method_problem"]:
        if amount > 2000:
            action = "payment_link"
            reason = "Customer action/card issue detected. Sending alternative payment link."
            confidence = 0.82
            benefit = "Enables customer to update payment method or choose alternative UPI/Netbanking."
        else:
            action = "reminder"
            reason = "Low-value transaction with customer action required. Sending friendly reminder."
            confidence = 0.78
            benefit = "Gentle nudge without high messaging cost."
    elif amount > 25000:
        action = "escalate"
        reason = "High transaction value (> ₹25,000). Escalating for human merchant review."
        confidence = 0.95
        benefit = "Ensures high-value subscriber receives personalized touchpoint."
    else:
        action = "retry"
        reason = "Standard retry recommendation based on moderate recovery probability."
        confidence = 0.75
        benefit = "Standard payment recovery attempt."

    state["recommended_action"] = action
    state["recommendation_reason"] = reason
    state["recommendation_confidence"] = confidence

    return state
