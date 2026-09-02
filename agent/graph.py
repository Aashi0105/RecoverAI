"""
LangGraph Revenue Recovery Agent Graph Assembly & Execution.

Orchestrates context loading, ML prediction, failure diagnosis, LLM strategy recommendation,
deterministic policy guard enforcement, mock action execution, outcome verification, and audit logging.
"""

import argparse
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes.context import load_context
from agent.nodes.prediction import predict_recovery
from agent.nodes.diagnosis import diagnose_failure
from agent.nodes.recommendation import recommend_action
from agent.nodes.policy import policy_guard
from agent.tools.mock_actions import execute_mock_action
from agent.nodes.verification import verify_outcome
from agent.nodes.audit import create_audit_log
from agent.demo_data import build_test_transaction


def route_after_policy(state: AgentState) -> str:
    """
    Conditional Edge Router:
    - ACT / APPROVED -> execute_mock_action
    - REFUSE / ESCALATE -> create_audit_log (skips execution entirely)
    """
    decision = state.get("policy_decision", "REFUSE")
    if decision in ["ACT", "APPROVED"]:
        return "execute_mock_action"
    else:
        return "create_audit_log"



def build_recovery_graph():
    """
    Constructs and compiles the complete LangGraph StateGraph.
    """
    workflow = StateGraph(AgentState)

    # 1. Add all workflow nodes
    workflow.add_node("load_context", load_context)
    workflow.add_node("predict_recovery", predict_recovery)
    workflow.add_node("diagnose_failure", diagnose_failure)
    workflow.add_node("recommend_action", recommend_action)
    workflow.add_node("policy_guard", policy_guard)
    workflow.add_node("execute_mock_action", execute_mock_action)
    workflow.add_node("verify_outcome", verify_outcome)
    workflow.add_node("create_audit_log", create_audit_log)

    # 2. Define standard sequential edges
    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "predict_recovery")
    workflow.add_edge("predict_recovery", "diagnose_failure")
    workflow.add_edge("diagnose_failure", "recommend_action")
    workflow.add_edge("recommend_action", "policy_guard")

    # 3. Add conditional edge after policy_guard
    workflow.add_conditional_edges(
        "policy_guard",
        route_after_policy,
        {
            "execute_mock_action": "execute_mock_action",
            "create_audit_log": "create_audit_log"
        }
    )

    # 4. Define post-execution sequence
    workflow.add_edge("execute_mock_action", "verify_outcome")
    workflow.add_edge("verify_outcome", "create_audit_log")
    workflow.add_edge("create_audit_log", END)

    app = workflow.compile()
    return app


# Compiled graph singleton instance
_compiled_agent = None


def get_recovery_agent():
    """Returns singleton compiled agent graph."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = build_recovery_graph()
    return _compiled_agent


def run_agent(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main invocation helper for the Revenue Recovery Agent.

    Input: Raw or failed transaction dictionary.
    Output: Clean structured summary result dictionary.
    """
    agent = get_recovery_agent()
    final_state = agent.invoke(transaction)

    # Format user-facing structured summary
    res = {
        "transaction_id": final_state.get("transaction_id"),
        "amount": final_state.get("amount"),
        "currency": final_state.get("currency", "INR"),
        "failure_reason": final_state.get("failure_reason"),
        "failure_category": final_state.get("failure_category"),
        "recovery_probability": final_state.get("recovery_probability"),
        "expected_recovery_value": final_state.get("expected_recovery_value"),
        "diagnosis": final_state.get("failure_diagnosis", {}).get("diagnosis"),
        "recommended_action": final_state.get("recommended_action"),
        "policy_decision": final_state.get("policy_decision"),
        "policy_reason": final_state.get("policy_reason"),
        "policy_violations": final_state.get("policy_violations", []),
        "decision_explanation": final_state.get("decision_explanation"),
        "selected_action": final_state.get("selected_action"),
        "action_status": final_state.get("action_result", {}).get("status", "not_executed"),
        "action_reference": final_state.get("action_result", {}).get("reference_id"),
        "verification_status": final_state.get("verification_result", {}).get("verification_status", "not_executed"),
        "money_recovered": final_state.get("money_recovered", 0.0),
        "agent_status": final_state.get("agent_status"),
        "audit_event": final_state.get("audit_event"),
        "diagnosis_source": final_state.get("diagnosis_source", "heuristic"),
        "recommendation_source": final_state.get("recommendation_source", "heuristic"),
        "recommendation_factors": final_state.get("recommendation_factors", [])
    }
    return res


def run_agent_demo():
    """
    Interactive demonstration runner executing 4 scenario traces using the transaction factory:
    1. Approved Transient Retry
    2. Retry Limit Exceeded (Blocked)
    3. High Value Transaction (Human Approval Required)
    4. Risk Related Failure (Blocked)
    """
    print("\n" + "=" * 75)
    print(" 🤖 RECOVERAI LANGGRAPH REVENUE RECOVERY AGENT DEMO TRACE")
    print("=" * 75)

    test_cases = [
        {
            "name": "SCENARIO 1: High-Confidence Transient Failure (Approved)",
            "txn": build_test_transaction(
                transaction_id="txn_demo_001",
                customer_id="cust_101",
                merchant_id="merch_01",
                amount=2500.0,
                failure_reason="network_timeout",
                failure_category="transient",
                customer_historical_success_rate=0.95,
                customer_previous_transactions=10,
                previous_failures_24h=0,
                recovery_attempt_count=0,
                ip_risk_score=0.05,
                velocity_score=0.10
            )
        },
        {
            "name": "SCENARIO 2: Retry Limit Exceeded (Blocked)",
            "txn": build_test_transaction(
                transaction_id="txn_demo_002",
                customer_id="cust_102",
                merchant_id="merch_01",
                amount=1500.0,
                failure_reason="network_timeout",
                failure_category="transient",
                previous_failures_24h=2,
                previous_failures_7d=2,
                recovery_attempt_count=2,
                customer_contacted_today=0
            )
        },
        {
            "name": "SCENARIO 3: High Value Transaction (Requires Human Approval)",
            "txn": build_test_transaction(
                transaction_id="txn_demo_003",
                customer_id="cust_103",
                merchant_id="merch_02",
                amount=45000.0,  # Exceeds ₹25,000 auto limit!
                failure_reason="insufficient_funds",
                failure_category="customer_action_required",
                customer_historical_success_rate=0.95,
                customer_previous_transactions=15,
                previous_failures_24h=0
            )
        },
        {
            "name": "SCENARIO 4: High IP Risk / Suspicious Transaction (Blocked)",
            "txn": build_test_transaction(
                transaction_id="txn_demo_004",
                customer_id="cust_104",
                merchant_id="merch_03",
                amount=8000.0,
                failure_reason="suspected_risk",
                failure_category="risk_related",
                ip_risk_score=0.85,
                velocity_score=0.75,
                device_changed=1,
                location_changed=1
            )
        }
    ]

    for case in test_cases:
        print("\n" + "-" * 75)
        print(f"📌 {case['name']}")
        print("-" * 75)
        res = run_agent(case["txn"])
        
        print(f" PAYMENT DETECTED   : {res['transaction_id']} (₹{res['amount']:,.2f} | Reason: {res['failure_reason']})")
        print(f" ML PROBABILITY     : {res['recovery_probability']:.4f} (Expected Value: ₹{res['expected_recovery_value']:,.2f})")
        print(f" FAILURE DIAGNOSIS  : {res['failure_category'].upper()} -> {res['diagnosis']}")
        print(f" RECOMMENDATION     : {res['recommended_action'].upper()}")
        print(f" POLICY DECISION    : {res['policy_decision']}")
        print(f" POLICY REASON      : {res['policy_reason']}")
        if res['policy_violations']:
            print(f" POLICY VIOLATIONS  : {res['policy_violations']}")
        print(f" ACTION STATUS      : {res['action_status'].upper()} (Selected: {res['selected_action']}, Ref: {res['action_reference']})")
        print(f" VERIFICATION       : {res['verification_status']} (Money Recovered: ₹{res['money_recovered']:,.2f})")
        print(f" AGENT STATUS       : {res['agent_status']}")

    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecoverAI Revenue Recovery Agent")
    parser.add_argument("--demo", action="store_true", help="Run interactive demonstration traces")
    args = parser.parse_args()

    if args.demo:
        run_agent_demo()
    else:
        print("LangGraph Revenue Recovery Agent compiled successfully. Use --demo flag to run sample traces.")
