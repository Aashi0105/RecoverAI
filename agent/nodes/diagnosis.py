"""
Node 3: Failure Diagnosis
Performs structured classification and diagnostic analysis of payment failure.
"""

from typing import Dict, Any
from agent.state import AgentState


def diagnose_failure(state: AgentState) -> AgentState:
    """
    Classifies failure root cause, severity, and customer action requirements.
    Uses deterministic failure mapping (or LLM structured output if available).
    """
    category = state.get("failure_category", "transient")
    reason = state.get("failure_reason", "network_timeout")
    
    # Deterministic diagnostic logic based on failure category
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

    diag_dict: Dict[str, Any] = {
        "failure_category": category,
        "reason": reason,
        "diagnosis": diagnosis,
        "severity": severity,
        "customer_action_required": cust_act
    }

    state["failure_diagnosis"] = diag_dict
    return state
