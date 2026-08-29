"""
Node 6: Mock Action Tools
Safe simulated recovery tools (retry, payment link, reminder, escalation).
Does NOT call live Razorpay APIs or claim money recovered directly.
"""

import uuid
from typing import Dict, Any
from agent.state import AgentState


def mock_retry_payment(transaction_id: str, amount: float) -> Dict[str, Any]:
    """Simulates initiating a smart payment retry request."""
    ref_id = f"MOCK_RETRY_{uuid.uuid4().hex[:8].upper()}"
    return {
        "status": "executed",
        "action": "retry",
        "reference_id": ref_id,
        "message": f"Mock smart retry submitted for transaction {transaction_id} (₹{amount:,.2f})"
    }


def mock_create_payment_link(transaction_id: str, amount: float, customer_id: str) -> Dict[str, Any]:
    """Simulates creating a Razorpay-style payment link."""
    ref_id = f"MOCK_PL_{uuid.uuid4().hex[:8].upper()}"
    return {
        "status": "executed",
        "action": "payment_link",
        "reference_id": ref_id,
        "payment_link_url": f"https://mock.razorpay.com/pl/{ref_id.lower()}",
        "message": f"Mock payment link created for customer {customer_id} (₹{amount:,.2f})"
    }


def mock_send_reminder(transaction_id: str, customer_id: str) -> Dict[str, Any]:
    """Simulates sending a customer SMS/email payment nudge."""
    ref_id = f"MOCK_NUDGE_{uuid.uuid4().hex[:8].upper()}"
    return {
        "status": "executed",
        "action": "reminder",
        "reference_id": ref_id,
        "message": f"Mock payment reminder notification queued for customer {customer_id}"
    }


def mock_escalate_to_human(transaction_id: str, reason: str) -> Dict[str, Any]:
    """Simulates escalating high-value or high-risk case to merchant desk."""
    ref_id = f"MOCK_ESC_{uuid.uuid4().hex[:8].upper()}"
    return {
        "status": "executed",
        "action": "escalate",
        "reference_id": ref_id,
        "message": f"Mock escalation ticket created for merchant review: {reason}"
    }


from backend.config import settings
from agent.tools.razorpay_actions import (
    razorpay_create_payment_link,
    razorpay_create_recovery_order
)


def execute_mock_action(state: AgentState) -> AgentState:
    """
    Executes recovery action corresponding to state['selected_action'].
    Only called when policy_decision == 'APPROVED'.
    Routes to Razorpay Test Mode API if RAZORPAY_ENABLED is True, otherwise uses mock tools.
    """
    action = state.get("selected_action")
    txn_id = state.get("transaction_id", "txn_000")
    amount = float(state.get("amount", 0.0))
    cust_id = state.get("customer_id", "cust_000")
    reason = state.get("policy_reason", "Action approved")

    use_razorpay = getattr(settings, "RAZORPAY_ENABLED", False)

    if use_razorpay:
        if action == "retry":
            result = razorpay_create_recovery_order(txn_id, amount)
        elif action == "payment_link":
            result = razorpay_create_payment_link(txn_id, amount, cust_id)
        elif action == "reminder":
            result = mock_send_reminder(txn_id, cust_id)
        elif action == "escalate":
            result = mock_escalate_to_human(txn_id, reason)
        else:
            result = {
                "status": "skipped",
                "action": "none",
                "reference_id": "NONE",
                "message": "No action executed."
            }
    else:
        if action == "retry":
            result = mock_retry_payment(txn_id, amount)
        elif action == "payment_link":
            result = mock_create_payment_link(txn_id, amount, cust_id)
        elif action == "reminder":
            result = mock_send_reminder(txn_id, cust_id)
        elif action == "escalate":
            result = mock_escalate_to_human(txn_id, reason)
        else:
            result = {
                "status": "skipped",
                "action": "none",
                "reference_id": "NONE",
                "message": "No mock action executed."
            }

    state["action_result"] = result
    state["agent_status"] = "EXECUTED"
    return state
