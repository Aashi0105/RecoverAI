"""
Node 5 & Core Engine: Deterministic Safety & Policy Decision Engine for RecoverAI.

Enforces strict financial guardrails, hard refusal rules, and escalation boundaries.
This engine sits after ML prediction and produces 3 explicit outcomes:
1. ACT      - Transaction is safe and eligible for automated recovery action (retry, payment_link, reminder).
2. ESCALATE - Autonomous action prohibited; requires human merchant review (high-value, uncertainty boundary, risk warning).
3. REFUSE   - System deliberately takes no automated action (fraud/risk, permanent card failure, streak limit, P < 0.35).
"""

import json
import os
from typing import Dict, Any, List, Tuple
import yaml

from agent.state import AgentState

DEFAULT_POLICY_PATH = os.path.join("evaluation", "business_policy.json")
DEFAULT_YAML_PATH = os.path.join("policies", "recovery_policy.yaml")

# High-value transaction threshold based on 95th percentile of amount distribution (INR)
HIGH_VALUE_TRANSACTION_THRESHOLD: float = 8500.00

# Boundary uncertainty region around frozen decision threshold Tau = 0.35
UNCERTAINTY_BAND_LOW: float = 0.32
UNCERTAINTY_BAND_HIGH: float = 0.38

ACTION_COSTS: Dict[str, float] = {
    "retry": 5.0,
    "reminder": 2.0,
    "payment_link": 12.0,
    "no_action": 0.0
}


def load_frozen_policy(policy_path: str = DEFAULT_POLICY_PATH) -> Dict[str, Any]:
    """
    Loads frozen business policy JSON payload.
    """
    if os.path.exists(policy_path):
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "policy_version": "1.0",
        "selection_dataset": "development_oof_5fold_cv",
        "selected_threshold": 0.35,
        "action_costs": ACTION_COSTS,
        "constraints": {"min_probability_threshold": 0.35}
    }


def map_recommended_action(failure_reason: str) -> Tuple[str, float]:
    """
    Maps failure_reason to recommended recovery action and associated cost.
    """
    reason = str(failure_reason).lower()
    if reason in ["network_timeout", "technical_error"]:
        action = "retry"
    elif reason in ["insufficient_funds", "authentication_failed", "limit_exceeded"]:
        action = "payment_link"
    elif reason in ["bank_declined", "customer_cancelled"]:
        action = "reminder"
    else:
        action = "no_action"

    return action, ACTION_COSTS[action]


def evaluate_transaction_policy(
    txn_data: Dict[str, Any],
    pred_prob: float,
    policy_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluates transaction data and ML predicted recovery probability against deterministic policy rules.
    Outputs structured decision payload containing decision ('ACT', 'ESCALATE', 'REFUSE'),
    recommended action, triggered rules, and economic justification.
    """
    if policy_config is None:
        policy_config = load_frozen_policy()

    tau = policy_config.get("selected_threshold", 0.35)
    policy_version = policy_config.get("policy_version", "1.0")

    txn_id = txn_data.get("transaction_id", "txn_unknown")
    timestamp = txn_data.get("timestamp", "N/A")
    cust_id = txn_data.get("customer_id", "cust_unknown")
    amount = float(txn_data.get("amount", 0.0))
    reason = str(txn_data.get("failure_reason", "unknown")).lower()
    category = str(txn_data.get("failure_category", "unknown")).lower()
    ip_risk = float(txn_data.get("ip_risk_score", 0.0))
    velocity = float(txn_data.get("velocity_score", 0.0))
    streak = int(txn_data.get("consecutive_failure_streak", 0))

    triggered_rules: List[str] = []
    decision: str = "ACT"
    justification: str = ""

    rec_action, action_cost = map_recommended_action(reason)

    # -------------------------------------------------------------------------
    # STAGE 1: HARD SAFETY REFUSAL RULES (Highest Priority)
    # -------------------------------------------------------------------------
    if category == "risk_related" or reason == "suspected_risk" or ip_risk > 0.70:
        decision = "REFUSE"
        rec_action = "no_action"
        action_cost = 0.0
        triggered_rules.append(f"HARD_SAFETY_REFUSE: Suspected fraud/risk failure (reason='{reason}', ip_risk={ip_risk:.2f})")

    elif category == "payment_method_problem" or reason in ["invalid_card", "card_expired"]:
        decision = "REFUSE"
        rec_action = "no_action"
        action_cost = 0.0
        triggered_rules.append(f"HARD_SAFETY_REFUSE: Permanent instrument failure (reason='{reason}')")

    elif streak >= 4:
        decision = "REFUSE"
        rec_action = "no_action"
        action_cost = 0.0
        triggered_rules.append(f"HARD_SAFETY_REFUSE: Consecutive failure streak limit reached (streak={streak} >= 4)")

    elif pred_prob < tau:
        decision = "REFUSE"
        rec_action = "no_action"
        action_cost = 0.0
        triggered_rules.append(f"OPERATIONAL_REFUSE: Recovery probability ({pred_prob:.4f}) below operational threshold (Tau={tau:.2f})")

    # -------------------------------------------------------------------------
    # STAGE 2: ESCALATION RULES (If not refused by hard safety rules)
    # -------------------------------------------------------------------------
    if decision != "REFUSE":
        if amount >= HIGH_VALUE_TRANSACTION_THRESHOLD:
            decision = "ESCALATE"
            triggered_rules.append(f"HIGH_VALUE_ESCALATE: Amount (₹{amount:,.2f}) exceeds high-value threshold (₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f})")

        if UNCERTAINTY_BAND_LOW <= pred_prob <= UNCERTAINTY_BAND_HIGH:
            decision = "ESCALATE"
            triggered_rules.append(f"BOUNDARY_UNCERTAINTY_ESCALATE: Probability ({pred_prob:.4f}) is within uncertainty band [{UNCERTAINTY_BAND_LOW:.2f}, {UNCERTAINTY_BAND_HIGH:.2f}]")

        if velocity > 0.65 and ip_risk > 0.50:
            decision = "ESCALATE"
            triggered_rules.append(f"RISK_WARNING_ESCALATE: Velocity score ({velocity:.2f}) and IP risk ({ip_risk:.2f}) require human review")

    # -------------------------------------------------------------------------
    # STAGE 3: ECONOMIC & POLICY JUSTIFICATION GENERATION
    # -------------------------------------------------------------------------
    if decision == "ACT":
        exp_gross = pred_prob * amount
        exp_net = exp_gross - action_cost
        justification = (
            f"Automated recovery action '{rec_action}' approved. Predicted recovery probability ({pred_prob:.4f}) "
            f"meets policy threshold ({tau:.2f}). Expected Net Value: ₹{exp_net:,.2f} (Gross ₹{exp_gross:,.2f} - Cost ₹{action_cost:.2f})."
        )
    elif decision == "ESCALATE":
        justification = (
            f"Automated execution paused. Transaction requires human merchant review due to: " + "; ".join(triggered_rules)
        )
    else: # REFUSE
        justification = (
            f"Automated recovery deliberately refused to prevent unnecessary cost or risk: " + "; ".join(triggered_rules)
        )

    explanation = build_decision_explanation(
        decision=decision,
        justification=justification,
        triggered_rules=triggered_rules,
        txn_data=txn_data,
        pred_prob=pred_prob,
        tau=tau,
        action_cost=action_cost,
        rec_action=rec_action
    )

    return {
        "transaction_id": txn_id,
        "timestamp": timestamp,
        "customer_id": cust_id,
        "amount": round(amount, 2),
        "failure_reason": reason,
        "failure_category": category,
        "recovery_probability": round(float(pred_prob), 4),
        "decision": decision,
        "recommended_action": rec_action,
        "action_cost": round(action_cost, 2),
        "triggered_rules": triggered_rules,
        "justification": justification,
        "policy_version": policy_version,
        "decision_explanation": explanation
    }


def build_decision_explanation(
    decision: str,
    justification: str,
    triggered_rules: List[str],
    txn_data: Dict[str, Any],
    pred_prob: float,
    tau: float,
    action_cost: float,
    rec_action: str
) -> Dict[str, Any]:
    """
    Constructs a structured, deterministic explanation of why the policy engine
    reached its final decision (ACT, ESCALATE, or REFUSE).

    Uses exclusively real evaluated inputs, existing frozen thresholds, and actual
    triggered rules. Zero fabrication or post-hoc hallucinations.
    """
    amount = float(txn_data.get("amount", 0.0)) if txn_data.get("amount") is not None else None
    reason = str(txn_data.get("failure_reason", "unknown")).lower()
    category = str(txn_data.get("failure_category", "unknown")).lower()
    ip_risk = float(txn_data.get("ip_risk_score", 0.0)) if txn_data.get("ip_risk_score") is not None else None
    velocity = float(txn_data.get("velocity_score", 0.0)) if txn_data.get("velocity_score") is not None else None
    streak = int(txn_data.get("consecutive_failure_streak", 0)) if txn_data.get("consecutive_failure_streak") is not None else 0

    # 1. Primary Factor Determination (Derived directly from triggered rule or approval)
    primary_factor = "STANDARD_POLICY_APPROVAL"
    if decision == "REFUSE":
        if any("Suspected fraud/risk" in r for r in triggered_rules):
            primary_factor = "HARD_SAFETY_FRAUD"
        elif any("Permanent instrument failure" in r for r in triggered_rules):
            primary_factor = "INSTRUMENT_FAILURE"
        elif any("Consecutive failure streak" in r for r in triggered_rules):
            primary_factor = "STREAK_LIMIT_EXCEEDED"
        elif any("below operational threshold" in r for r in triggered_rules):
            primary_factor = "PROBABILITY_BELOW_THRESHOLD"
        else:
            primary_factor = "SAFETY_POLICY_REFUSAL"
    elif decision == "ESCALATE":
        if any("exceeds high-value threshold" in r for r in triggered_rules):
            primary_factor = "HIGH_VALUE_THRESHOLD"
        elif any("uncertainty band" in r for r in triggered_rules):
            primary_factor = "UNCERTAINTY_BAND"
        elif any("Velocity score" in r for r in triggered_rules):
            primary_factor = "VELOCITY_RISK"
        else:
            primary_factor = "POLICY_ESCALATION"

    # 2. Structured Policy Checks Breakdown
    # Checks evaluate actual conditions; status must be one of: PASSED, FAILED, ESCALATED, UNKNOWN
    fraud_failed = bool(category == "risk_related" or reason == "suspected_risk" or (ip_risk is not None and ip_risk > 0.70))
    instrument_failed = bool(category == "payment_method_problem" or reason in ["invalid_card", "card_expired"])
    streak_failed = bool(streak >= 4)
    viability_failed = bool(pred_prob is not None and pred_prob < tau)

    val_escalated = bool(decision == "ESCALATE" and amount is not None and amount >= HIGH_VALUE_TRANSACTION_THRESHOLD)
    conf_escalated = bool(decision == "ESCALATE" and pred_prob is not None and UNCERTAINTY_BAND_LOW <= pred_prob <= UNCERTAINTY_BAND_HIGH)
    vel_escalated = bool(decision == "ESCALATE" and velocity is not None and ip_risk is not None and velocity > 0.65 and ip_risk > 0.50)

    policy_checks = {
        "fraud_risk": "FAILED" if fraud_failed else "PASSED",
        "instrument_status": "FAILED" if instrument_failed else "PASSED",
        "failure_streak": "FAILED" if streak_failed else "PASSED",
        "recovery_viability": "FAILED" if viability_failed else "PASSED",
        "value_threshold": "ESCALATED" if val_escalated else "PASSED",
        "confidence_band": "ESCALATED" if conf_escalated else "PASSED",
        "velocity_risk": "ESCALATED" if vel_escalated else "PASSED"
    }

    # 3. Deterministic Reasons Composition
    reasons: List[str] = []
    if decision == "ACT":
        if pred_prob is not None:
            reasons.append(f"Predicted recovery probability ({pred_prob:.1%}) meets or exceeds operational threshold (τ = {tau:.2f}).")
        reasons.append(f"Failure categorized as transient ({category}) with no permanent instrument defect.")
        reasons.append(f"Consecutive failure streak ({streak}) is within safe retry limit (< 4).")
        if amount is not None:
            reasons.append(f"Transaction amount (₹{amount:,.2f}) is within autonomous execution boundary (< ₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}).")
        reasons.append("Zero safety policy violations detected.")
    elif decision == "REFUSE":
        for rule in triggered_rules:
            if "Suspected fraud/risk" in rule:
                risk_str = f"{ip_risk:.2f}" if ip_risk is not None else "N/A"
                reasons.append(f"High risk score detected (reason='{reason}', IP risk={risk_str} > 0.70). Recovery blocked to prevent fraud.")
            elif "Permanent instrument failure" in rule:
                reasons.append(f"Permanent payment instrument failure (reason='{reason}'). Retrying will not succeed.")
            elif "Consecutive failure streak" in rule:
                reasons.append(f"Consecutive failure streak limit reached ({streak} >= 4). Automatic retries halted to prevent fee churn.")
            elif "below operational threshold" in rule:
                prob_str = f"{pred_prob:.1%}" if pred_prob is not None else "N/A"
                reasons.append(f"Recovery probability ({prob_str}) is below operational viability threshold (τ = {tau:.2f}). Expected recovery value does not justify action cost.")
            else:
                reasons.append(rule)
        if not reasons:
            reasons.append("Deliberately refused by safety guardrails to prevent unnecessary cost or risk.")
    else:  # ESCALATE
        for rule in triggered_rules:
            if "Amount" in rule and "high-value" in rule:
                amt_str = f"₹{amount:,.2f}" if amount is not None else "N/A"
                reasons.append(f"Transaction amount ({amt_str}) exceeds autonomous limit (₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}). Human merchant review required.")
            elif "uncertainty band" in rule:
                prob_str = f"{pred_prob:.1%}" if pred_prob is not None else "N/A"
                reasons.append(f"Recovery probability ({prob_str}) falls in boundary uncertainty band [{UNCERTAINTY_BAND_LOW:.2f}, {UNCERTAINTY_BAND_HIGH:.2f}]. Model confidence requires human sign-off.")
            elif "Velocity score" in rule:
                vel_str = f"{velocity:.2f}" if velocity is not None else "N/A"
                risk_str = f"{ip_risk:.2f}" if ip_risk is not None else "N/A"
                reasons.append(f"Elevated velocity score ({vel_str}) and IP risk ({risk_str}) require human review before action.")
            else:
                reasons.append(rule)
        if not reasons:
            reasons.append("Escalated to Merchant Review Queue for human approval.")

    # 4. Metrics Evaluated
    metrics_evaluated = {
        "recovery_probability": round(float(pred_prob), 4) if pred_prob is not None else None,
        "operational_threshold_tau": round(float(tau), 4) if tau is not None else None,
        "transaction_amount": round(float(amount), 2) if amount is not None else None,
        "high_value_limit": round(float(HIGH_VALUE_TRANSACTION_THRESHOLD), 2),
        "consecutive_failure_streak": int(streak) if streak is not None else None,
        "streak_limit": 4,
        "ip_risk_score": round(float(ip_risk), 4) if ip_risk is not None else None,
        "ip_risk_limit": 0.70
    }

    return {
        "decision": decision,
        "summary": justification,
        "primary_factor": primary_factor,
        "reasons": reasons,
        "policy_checks": policy_checks,
        "metrics_evaluated": metrics_evaluated
    }


def policy_guard(state: AgentState) -> AgentState:
    """
    LangGraph Node integration wrapper for AgentState backwards compatibility.
    """
    txn_data = {
        "transaction_id": state.get("transaction_id", "txn_unknown"),
        "customer_id": state.get("customer_id", "cust_unknown"),
        "amount": state.get("amount", 0.0),
        "failure_reason": state.get("failure_reason", "unknown"),
        "failure_category": state.get("failure_category", "unknown"),
        "ip_risk_score": state.get("customer_context", {}).get("ip_risk_score", state.get("ip_risk_score", 0.0)),
        "velocity_score": state.get("customer_context", {}).get("velocity_score", state.get("velocity_score", 0.0)),
        "consecutive_failure_streak": state.get("customer_context", {}).get("consecutive_failure_streak", state.get("consecutive_failure_streak", 0))
    }

    pred_prob = state.get("recovery_probability", 0.0)
    policy_eval = evaluate_transaction_policy(txn_data, pred_prob)

    state["policy_decision"] = policy_eval["decision"]
    state["policy_reason"] = policy_eval["justification"]
    state["policy_violations"] = policy_eval["triggered_rules"]
    state["decision_explanation"] = policy_eval.get("decision_explanation")
    state["selected_action"] = policy_eval["recommended_action"] if policy_eval["decision"] == "ACT" else None
    
    if policy_eval["decision"] == "ACT":
        state["agent_status"] = "APPROVED"
    elif policy_eval["decision"] == "ESCALATE":
        state["agent_status"] = "AWAITING_APPROVAL"
    else:
        state["agent_status"] = "BLOCKED"

    return state
