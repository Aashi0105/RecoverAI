"""
Node 7: Outcome Verification
Simulates outcome verification of executed mock recovery actions.
Separates action execution from actual financial recovery verification.
"""

import numpy as np
from agent.state import AgentState


def verify_outcome(state: AgentState) -> AgentState:
    """
    Verifies whether the executed mock action successfully recovered the payment.
    """
    action_res = state.get("action_result", {})
    action_status = action_res.get("status")
    selected_action = state.get("selected_action")
    amount = float(state.get("amount", 0.0))
    prob = float(state.get("recovery_probability", 0.50))
    category = state.get("failure_category", "transient")

    if action_status != "executed":
        ver_res = {
            "verification_status": "PENDING",
            "payment_recovered": False,
            "money_recovered": 0.0,
            "result_reason": "No action was executed to verify."
        }
        state["verification_result"] = ver_res
        state["money_recovered"] = 0.0
        state["agent_status"] = "VERIFIED"
        return state

    if selected_action == "escalate":
        ver_res = {
            "verification_status": "ESCALATED",
            "payment_recovered": False,
            "money_recovered": 0.0,
            "result_reason": "Case escalated to merchant human review. Outcome pending."
        }
        state["verification_result"] = ver_res
        state["money_recovered"] = 0.0
        state["agent_status"] = "VERIFIED"
        return state

    # Real Razorpay Provider Verification
    if action_res.get("provider") == "razorpay":
        prov_status = action_res.get("provider_status", "created")
        if prov_status in ["paid", "captured"]:
            ver_status = "SUCCESS"
            is_success = True
            money_rec = amount
            reason = f"Razorpay payment confirmed as captured for action '{selected_action}'."
        elif prov_status in ["failed", "expired"]:
            ver_status = "FAILED"
            is_success = False
            money_rec = 0.0
            reason = f"Razorpay action '{selected_action}' failed or link expired."
        else:
            # 'created', 'issued', or pending status
            ver_status = "PENDING"
            is_success = False
            money_rec = 0.0
            reason = f"Razorpay {selected_action} created ({action_res.get('reference_id')}). Awaiting customer payment capture."

        ver_res = {
            "verification_status": ver_status,
            "payment_recovered": is_success,
            "money_recovered": money_rec,
            "result_reason": reason
        }
        state["verification_result"] = ver_res
        state["money_recovered"] = money_rec
        state["agent_status"] = "VERIFIED"
        return state

    # Simulated verification logic based on failure category and probability
    # High probability & transient -> high chance of simulated success
    if category == "transient" and prob >= 0.70:
        is_success = True
    elif category == "payment_method_problem" and selected_action == "retry":
        is_success = False  # Expired card retries always fail
    else:
        # Probabilistic simulation
        is_success = bool(prob >= 0.60)

    if is_success:
        ver_status = "SUCCESS"
        money_rec = amount
        reason = f"Simulated recovery verified successfully for action '{selected_action}'."
    else:
        ver_status = "FAILED"
        money_rec = 0.0
        reason = f"Simulated recovery attempt failed for action '{selected_action}'."

    ver_res = {
        "verification_status": ver_status,
        "payment_recovered": is_success,
        "money_recovered": money_rec,
        "result_reason": reason
    }

    state["verification_result"] = ver_res
    state["money_recovered"] = money_rec
    state["agent_status"] = "VERIFIED"
    return state
