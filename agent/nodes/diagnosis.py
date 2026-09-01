"""
Node 3: Failure Diagnosis
Performs structured classification and diagnostic analysis of payment failure.
Supports real LLM structured decision support with safe deterministic heuristic fallback.
"""

from typing import Dict, Any
from agent.state import AgentState
from agent.services.llm_service import is_llm_available, generate_llm_diagnosis


def heuristic_diagnosis(category: str, reason: str) -> Dict[str, Any]:
    """
    Deterministic diagnostic logic based on failure category and reason.
    Provides 100% reliable fallback when LLM is unconfigured, disabled, or offline.
    """
    if category == "transient":
        diagnosis = "Temporary gateway or network connectivity timeout between customer bank and merchant server."
        severity = "LOW"
        cust_act = False
    elif category == "technical":
        diagnosis = "Technical communication error or gateway processing failure during payment auth."
        severity = "MEDIUM"
        cust_act = False
    elif category == "customer_action_required":
        diagnosis = f"Action required by customer (Reason: {reason.replace('_', ' ')}). Payment limit or balance exceeded."
        severity = "MEDIUM"
        cust_act = True
    elif category == "payment_method_problem":
        diagnosis = f"Card or payment instrument issue (Reason: {reason.replace('_', ' ')}). Card expired or invalid."
        severity = "HIGH"
        cust_act = True
    elif category == "bank_decline":
        diagnosis = "Bank explicitly declined payment authorization without exposing specific reason."
        severity = "MEDIUM"
        cust_act = True
    elif category == "risk_related":
        diagnosis = "Transaction flagged by risk models due to suspicious IP, velocity, or fraud signal."
        severity = "HIGH"
        cust_act = False
    else:
        diagnosis = "Unclassified payment failure."
        severity = "MEDIUM"
        cust_act = False

    return {
        "failure_category": category,
        "reason": reason,
        "diagnosis": diagnosis,
        "severity": severity,
        "customer_action_required": cust_act,
        "key_factors": [reason.replace("_", " "), f"severity: {severity}"]
    }


def diagnose_failure(state: AgentState) -> AgentState:
    """
    Classifies failure root cause, severity, and customer action requirements.
    Uses LLM structured output if enabled and configured; otherwise falls back to heuristics.
    """
    category = state.get("failure_category", "transient")
    reason = state.get("failure_reason", "network_timeout")

    # 1. Try real LLM structured diagnosis if available
    if is_llm_available():
        context = {
            "transaction_id": state.get("transaction_id", "unknown"),
            "amount": state.get("amount", 0.0),
            "currency": state.get("currency", "INR"),
            "payment_method": state.get("payment_method", "unknown"),
            "failure_reason": reason,
            "failure_category": category,
            "previous_failures_24h": state.get("customer_context", {}).get("previous_failures_24h", state.get("previous_failures_24h", 0)),
            "previous_failures_7d": state.get("customer_context", {}).get("previous_failures_7d", state.get("previous_failures_7d", 0)),
            "consecutive_failure_streak": state.get("customer_context", {}).get("consecutive_failure_streak", state.get("consecutive_failure_streak", 0)),
            "ip_risk_score": state.get("customer_context", {}).get("ip_risk_score", state.get("ip_risk_score", 0.0)),
            "velocity_score": state.get("customer_context", {}).get("velocity_score", state.get("velocity_score", 0.0)),
            "recovery_probability": state.get("recovery_probability")
        }

        try:
            llm_result = generate_llm_diagnosis(context)
            if llm_result:
                state["failure_diagnosis"] = llm_result.model_dump()
                state["diagnosis_source"] = "llm"
                return state
        except Exception:
            pass  # Fall through safely to heuristic

    # 2. Deterministic Heuristic Fallback
    diag_dict = heuristic_diagnosis(category, reason)
    state["failure_diagnosis"] = diag_dict
    state["diagnosis_source"] = "heuristic"
    return state
