"""
Node 8: Audit Log
Constructs complete, immutable audit record of the agent's decision and execution trace.
"""

from datetime import datetime, timezone
from typing import Dict, Any
from agent.state import AgentState


def create_audit_log(state: AgentState) -> AgentState:
    """
    Assembles structured audit record capturing the entire decision chain:
    What was known? What did ML predict? What did LLM recommend?
    What did policy approve? What was executed? What was verified?
    """
    audit_record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": state.get("transaction_id"),
        "customer_id": state.get("customer_id"),
        "merchant_id": state.get("merchant_id"),
        "amount": state.get("amount"),
        "currency": state.get("currency", "INR"),
        
        # ML Predictions
        "recovery_probability": state.get("recovery_probability"),
        "expected_recovery_value": state.get("expected_recovery_value"),
        "risk_level": state.get("risk_level"),
        
        # Agent Diagnosis & Recommendation
        "failure_diagnosis": state.get("failure_diagnosis"),
        "recommended_action": state.get("recommended_action"),
        "recommendation_reason": state.get("recommendation_reason"),
        
        # Policy Guard Decision
        "policy_decision": state.get("policy_decision"),
        "policy_reason": state.get("policy_reason"),
        "policy_violations": state.get("policy_violations", []),
        
        # Execution & Verification
        "selected_action": state.get("selected_action"),
        "action_result": state.get("action_result"),
        "verification_result": state.get("verification_result"),
        "money_recovered": state.get("money_recovered", 0.0),
        
        "agent_status": state.get("agent_status", "COMPLETED")
    }

    state["audit_event"] = audit_record
    if state.get("agent_status") != "AWAITING_APPROVAL":
        state["agent_status"] = "COMPLETED"

    return state
