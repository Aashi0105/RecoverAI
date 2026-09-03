"""
Demo Simulator Router for RecoverAI REST API.

Provides endpoints for interactive 3-5 minute Razorpay Buildathon demonstrations:
1. POST /api/v1/demo/simulate — One-click simulation of 4 predefined scenario journeys.
2. POST /api/v1/demo/simulate-webhook — Closed-loop Razorpay webhook settlement simulation.
3. GET /api/v1/demo/scenarios — Predefined scenario metadata and expected behaviors.
"""

import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from database.database import get_db
from database.repository import (
    save_recovery_audit,
    get_approval_request,
    process_webhook_lifecycle_event
)
from payment.webhook import normalize_razorpay_webhook
from database.models import FailedPayment, ApprovalRequest, PaymentExecutionClaim
from agent.graph import run_agent
from agent.demo_data import build_test_transaction
from agent.nodes.policy import HIGH_VALUE_TRANSACTION_THRESHOLD
from backend.schemas.demo import (
    DemoSimulationRequest,
    DemoWebhookSimulationRequest,
    DemoSimulationResponse,
    DemoScenarioInfo
)

router = APIRouter()

PREDEFINED_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "auto_recovery": {
        "id": "auto_recovery",
        "name": "Scenario A — Smart Auto Recovery",
        "description": "Transient network timeout on a valued customer. Agent analyzes, ML predicts high recovery probability, Policy ACT approves, and recovery action executes automatically.",
        "expected_flow": "Analysis -> High ML Probability -> LLM Recommendation -> Policy ACT -> Auto Execution -> Verification",
        "expected_decision": "ACT",
        "expected_action": "retry",
        "amount": 2500.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "business_title": "Autonomous Revenue Recovery",
        "business_impact": "₹2,500 recovery action successfully initiated autonomously without requiring manual human intervention."
    },
    "human_approval": {
        "id": "human_approval",
        "name": "Scenario B — High-Value Human Approval",
        "description": f"Transaction amount (₹14,500) exceeds safety threshold (₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}). Policy halts automated execution and routes directly into the Merchant Approval Queue.",
        "expected_flow": "Analysis -> LLM Recommendation -> Policy ESCALATE -> Enters Approval Queue -> Merchant Decision",
        "expected_decision": "ESCALATE",
        "expected_action": "retry",
        "amount": 14500.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "business_title": "Controlled High-Value Recovery",
        "business_impact": f"₹14,500 high-value action required merchant approval. Autonomous execution was safely halted by policy guard."
    },
    "fraud_block": {
        "id": "fraud_block",
        "name": "Scenario C — Fraud / High Risk Block",
        "description": "Suspicious transaction with high IP risk (0.92) and velocity anomalies. Deterministic Policy Guard immediately blocks execution to protect the merchant from chargebacks. Human override is prohibited.",
        "expected_flow": "Analysis -> High IP Risk Detected -> Policy REFUSE -> Zero Actions Executed -> Safety Audit Recorded",
        "expected_decision": "REFUSE",
        "expected_action": "no_action",
        "amount": 7500.0,
        "failure_reason": "suspected_risk",
        "failure_category": "risk_related",
        "business_title": "Chargeback & Fraud Protection",
        "business_impact": "Potential recovery action was blocked due to critical IP fraud risk (0.92). Zero external money movement occurred."
    },
    "low_probability": {
        "id": "low_probability",
        "name": "Scenario D — Low Recovery Probability / Negative EV",
        "description": "Persistent bank decline with recovery probability below policy threshold (Tau = 0.35). System rationally refuses recovery to prevent burning action fees on hopeless attempts.",
        "expected_flow": "Analysis -> ML Probability < 0.35 -> Negative Expected Value -> Policy REFUSE -> Spend Protected",
        "expected_decision": "REFUSE",
        "expected_action": "no_action",
        "amount": 3200.0,
        "failure_reason": "bank_declined",
        "failure_category": "bank_decline",
        "business_title": "Negative EV & Fee Avoidance",
        "business_impact": "Low-probability recovery attempt was avoided. Customer spam and unnecessary action fees prevented."
    }
}


def construct_timeline(result: Dict[str, Any], scenario: str, closed_loop: bool = False) -> List[Dict[str, Any]]:
    """Constructs user-friendly step-by-step timeline cards for UI visualization."""
    amount = result.get("amount", 0.0)
    prob = result.get("recovery_probability", 0.0) or 0.0
    ev = result.get("expected_recovery_value", 0.0) or 0.0
    decision = result.get("policy_decision", "REFUSE")
    action = result.get("selected_action") or result.get("recommended_action") or "no_action"
    source_diag = result.get("diagnosis_source", "heuristic")
    source_rec = result.get("recommendation_source", "heuristic")

    steps = [
        {
            "step_number": 1,
            "title": "Payment Failure Detected",
            "icon": "",
            "summary": f"Transaction {result.get('transaction_id')} of ₹{amount:,.2f} failed via gateway.",
            "details": {
                "transaction_id": result.get("transaction_id"),
                "amount": f"₹{amount:,.2f}",
                "failure_reason": result.get("failure_reason"),
                "failure_category": result.get("failure_category"),
                "currency": result.get("currency", "INR")
            }
        },
        {
            "step_number": 2,
            "title": "ML Recovery Prediction",
            "icon": "",
            "summary": f"Estimated recovery probability: {prob * 100:.1f}%. Expected Recovery Value: ₹{ev:,.2f}.",
            "details": {
                "recovery_probability": f"{prob * 100:.1f}%",
                "expected_recovery_value": f"₹{ev:,.2f}",
                "model": "EXP_0 Logistic Regression (Calibrated)",
                "decision_threshold_tau": "0.35"
            }
        },
        {
            "step_number": 3,
            "title": f"Failure Diagnosis ({'LLM' if source_diag == 'llm' else 'Heuristic Fallback'})",
            "icon": "",
            "summary": result.get("diagnosis") or "Failure pattern analyzed.",
            "details": {
                "diagnosis": result.get("diagnosis"),
                "diagnosis_source": source_diag.upper(),
                "failure_category": result.get("failure_category")
            }
        },
        {
            "step_number": 4,
            "title": f"Recovery Recommendation ({'LLM' if source_rec == 'llm' else 'Heuristic Fallback'})",
            "icon": "",
            "summary": f"Strategy: {action.upper()}.",
            "details": {
                "recommended_action": action.upper(),
                "recommendation_source": source_rec.upper(),
                "contributing_factors": result.get("recommendation_factors", []),
                "expected_benefit": result.get("recommendation_expected_benefit") or "Maximizes probability of customer completion."
            }
        },
        {
            "step_number": 5,
            "title": "Deterministic Policy Decision",
            "icon": "",
            "summary": f"Policy Guard Verdict: {decision}.",
            "details": {
                "decision": decision,
                "policy_reason": result.get("policy_reason"),
                "policy_authority": "Deterministic Policy Guard (Immutable Rules)",
                "violations": result.get("policy_violations", [])
            }
        }
    ]

    # Step 6: Execution Layer
    if decision == "ACT":
        steps.append({
            "step_number": 6,
            "title": "Controlled Action Execution",
            "icon": "",
            "summary": f"Recovery action '{action}' executed safely.",
            "details": {
                "action": action,
                "status": result.get("action_status", "executed").upper(),
                "reference_id": result.get("action_reference") or "SIMULATED_REF",
                "execution_mode": "Safe Dry Run / Test Mode"
            }
        })
    elif decision == "ESCALATE":
        steps.append({
            "step_number": 6,
            "title": "Escalated to Merchant Review Queue",
            "icon": "",
            "summary": "Automated execution halted. Awaiting merchant human authorization.",
            "details": {
                "queue": "Merchant Approval Queue",
                "status": "PENDING_APPROVAL",
                "action_held": action,
                "reason": "Amount exceeds autonomous threshold"
            }
        })
    else:
        steps.append({
            "step_number": 6,
            "title": "Execution Blocked by Safety Guard",
            "icon": "",
            "summary": "No external action executed. Customer protected from spam or risk.",
            "details": {
                "status": "BLOCKED_FOR_SAFETY",
                "action_executed": None,
                "reason": result.get("policy_reason")
            }
        })

    # Step 7: Outcome / Verification
    if closed_loop and decision == "ACT":
        steps.append({
            "step_number": 7,
            "title": "Closed-Loop Settlement Confirmed",
            "icon": "",
            "summary": f"Customer completed payment link. Recovered: ₹{amount:,.2f}.",
            "details": {
                "verification_status": "SUCCESS",
                "money_recovered": f"₹{amount:,.2f}",
                "webhook_verified": True,
                "status": "RECOVERED"
            }
        })
    else:
        steps.append({
            "step_number": 7,
            "title": "Outcome Verification Status",
            "icon": "",
            "summary": f"Status: {result.get('agent_status', 'COMPLETED')}.",
            "details": {
                "verification_status": result.get("verification_status", "PENDING"),
                "money_recovered": f"₹{result.get('money_recovered', 0.0):,.2f}",
                "agent_status": result.get("agent_status")
            }
        })

    return steps


@router.get(
    "/scenarios",
    response_model=List[DemoScenarioInfo],
    summary="List Predefined Demo Scenarios",
    description="Returns metadata and expected behavior for all 4 Buildathon demo scenarios."
)
def get_demo_scenarios():
    return [DemoScenarioInfo(**data) for data in PREDEFINED_SCENARIOS.values()]


@router.post(
    "/simulate",
    response_model=DemoSimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Simulate Demo Scenario Journey",
    description="One-click execution of a demo scenario through the full RecoverAI agent workflow, persisting audit and queue records."
)
def simulate_demo_scenario(
    payload: DemoSimulationRequest,
    db: Session = Depends(get_db)
):
    scen_key = payload.scenario.lower()
    if scen_key not in PREDEFINED_SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario '{payload.scenario}'. Supported scenarios: {list(PREDEFINED_SCENARIOS.keys())}"
        )

    scen_info = PREDEFINED_SCENARIOS[scen_key]
    txn_id = f"demo_{scen_key[:4]}_{uuid.uuid4().hex[:6]}"
    amt = payload.amount if payload.amount is not None else scen_info["amount"]

    # 1. Build scenario-specific transaction context
    if scen_key == "auto_recovery":
        txn = build_test_transaction(
            transaction_id=txn_id,
            customer_id="cust_buildathon_01",
            merchant_id="merch_razorpay_demo",
            amount=amt,
            customer_average_transaction=amt,
            failure_reason="network_timeout",
            failure_category="transient",
            customer_historical_success_rate=0.95,
            customer_previous_transactions=12,
            previous_failures_24h=0,
            recovery_attempt_count=0,
            ip_risk_score=0.04,
            velocity_score=0.08
        )
    elif scen_key == "human_approval":
        txn = build_test_transaction(
            transaction_id=txn_id,
            customer_id="cust_enterprise_02",
            merchant_id="merch_razorpay_demo",
            amount=amt,  # > ₹8,500 threshold
            customer_average_transaction=amt,
            failure_reason="network_timeout",
            failure_category="transient",
            customer_historical_success_rate=0.95,
            customer_previous_transactions=15,
            previous_failures_24h=0,
            recovery_attempt_count=0,
            ip_risk_score=0.05,
            velocity_score=0.10
        )
    elif scen_key == "fraud_block":
        txn = build_test_transaction(
            transaction_id=txn_id,
            customer_id="cust_suspicious_03",
            merchant_id="merch_razorpay_demo",
            amount=amt,
            failure_reason="suspected_risk",
            failure_category="risk_related",
            ip_risk_score=0.92,  # Strict fraud flag
            velocity_score=0.85,
            customer_historical_success_rate=0.40,
            previous_failures_24h=2
        )
    elif scen_key == "low_probability":
        txn = build_test_transaction(
            transaction_id=txn_id,
            customer_id="cust_churned_04",
            merchant_id="merch_razorpay_demo",
            amount=amt,
            customer_average_transaction=1200.0,
            failure_reason="bank_declined",
            failure_category="bank_decline",
            payment_method="netbanking",
            payment_network="axis",
            payment_channel="web",
            customer_historical_success_rate=0.0,
            customer_previous_transactions=2,
            previous_failures_24h=2,
            recovery_attempt_count=2,
            ip_risk_score=0.30,
            velocity_score=0.85
        )
    else:
        txn = build_test_transaction(transaction_id=txn_id, amount=amt)

    # 2. Run full agent workflow
    result = run_agent(txn)

    # 3. Persist to database (Auto-queues if ESCALATE)
    save_recovery_audit(db, result)

    # 4. Optional: closed-loop settlement simulation
    closed_loop_applied = False
    if payload.simulate_closed_loop and result["policy_decision"] == "ACT":
        # Simulate customer payment
        payment = db.query(FailedPayment).filter(FailedPayment.id == txn_id).first()
        if payment:
            payment.status = "RECOVERED"
            db.commit()
        result["verification_status"] = "SUCCESS"
        result["money_recovered"] = amt
        result["agent_status"] = "RECOVERED"
        closed_loop_applied = True

    # 5. Build timeline
    timeline = construct_timeline(result, scen_key, closed_loop=closed_loop_applied)
    is_matched = bool(result.get("policy_decision") == scen_info.get("expected_decision"))

    return DemoSimulationResponse(
        scenario=scen_key,
        transaction_id=result["transaction_id"],
        customer_id=txn.get("customer_id", "cust_001"),
        merchant_id=txn.get("merchant_id", "merch_001"),
        amount=result["amount"],
        currency=result.get("currency", "INR"),
        failure_reason=result.get("failure_reason", "unknown"),
        failure_category=result.get("failure_category", "unknown"),
        recovery_probability=result.get("recovery_probability", 0.0),
        expected_recovery_value=result.get("expected_recovery_value", 0.0),
        diagnosis=result.get("diagnosis"),
        diagnosis_source=result.get("diagnosis_source", "heuristic"),
        recommended_action=result.get("recommended_action", "no_action"),
        recommendation_source=result.get("recommendation_source", "heuristic"),
        recommendation_confidence=0.85,
        recommendation_factors=result.get("recommendation_factors", []),
        recommendation_expected_benefit="Maximizes recovery probability based on past transaction patterns.",
        policy_decision=result.get("policy_decision", "REFUSE"),
        policy_reason=result.get("policy_reason", ""),
        policy_violations=result.get("policy_violations", []),
        expected_decision=scen_info.get("expected_decision", "ACT"),
        is_policy_matched=is_matched,
        business_title=scen_info.get("business_title", ""),
        business_impact=scen_info.get("business_impact", ""),
        action_status=result.get("action_status", "not_executed"),
        selected_action=result.get("selected_action"),
        action_reference=result.get("action_reference"),
        verification_status=result.get("verification_status", "not_executed"),
        money_recovered=result.get("money_recovered", 0.0),
        agent_status=result.get("agent_status", "COMPLETED"),
        timeline_steps=timeline,
        closed_loop_simulated=closed_loop_applied
    )


@router.post(
    "/simulate-webhook",
    status_code=status.HTTP_200_OK,
    summary="Simulate Closed-Loop Razorpay Webhook",
    description="Simulates incoming Razorpay webhook event (e.g. payment_link.paid) to close the recovery loop."
)
def simulate_webhook_event(
    payload: DemoWebhookSimulationRequest,
    db: Session = Depends(get_db)
):
    txn_id = payload.transaction_id
    event_type = payload.event

    # Ensure transaction exists
    payment = db.query(FailedPayment).filter(FailedPayment.id == txn_id).first()
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{txn_id}' was not found."
        )

    # Construct mock webhook payload
    plink_id = f"plink_demo_{txn_id[-6:]}"
    mock_payload = {
        "event": event_type,
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "reference_id": txn_id,
                    "status": "paid" if event_type == "payment_link.paid" else "expired",
                    "amount": int(payment.amount * 100),
                    "currency": "INR"
                }
            }
        }
    }

    # Ensure claim exists for correlation
    from database.repository import (
        get_execution_claim_by_payment_id,
        create_execution_claim,
        mark_execution_succeeded
    )
    claim = get_execution_claim_by_payment_id(db, txn_id)
    if not claim:
        create_execution_claim(
            db=db,
            idempotency_key=f"idemp_{txn_id}",
            payment_id=txn_id,
            action_type="payment_link",
            amount=payment.amount
        )
        mark_execution_succeeded(
            db=db,
            idempotency_key=f"idemp_{txn_id}",
            payment_link_id=plink_id,
            short_url=f"https://rzp.io/i/{txn_id[-6:]}"
        )

    # Process through closed-loop webhook engine
    event = normalize_razorpay_webhook(mock_payload)
    res = process_webhook_lifecycle_event(db, event)

    db.refresh(payment)
    return {
        "message": f"Webhook '{event_type}' processed successfully.",
        "transaction_id": txn_id,
        "payment_status": payment.status,
        "result": res
    }
