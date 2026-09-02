"""
Comprehensive Unit Tests for Feature 2: Structured Decision Explanation Layer.

Verifies:
1. Schema Integrity: all required keys exist and match types.
2. Decision Consistency: explanation['decision'] == policy_decision.
3. ACT Scenario: checks pass, reasons accurate, no false refusal flags.
4. REFUSE Scenarios: fraud/risk, low probability, instrument failure, streak limit.
5. ESCALATE Scenarios: high-value, boundary uncertainty, velocity risk.
6. Metric Traceability: metrics_evaluated matches transaction inputs and policy config.
"""

import pytest
from agent.graph import run_agent
from agent.nodes.policy import evaluate_transaction_policy, HIGH_VALUE_TRANSACTION_THRESHOLD
from agent.demo_data import build_test_transaction


def test_schema_integrity():
    """Test 1: Verify the explanation contains all required schema keys."""
    txn = build_test_transaction(
        transaction_id="txn_schema_test",
        amount=2000.0,
        failure_reason="network_timeout",
        failure_category="transient"
    )
    res = run_agent(txn)
    
    assert "decision_explanation" in res
    exp = res["decision_explanation"]
    assert isinstance(exp, dict)

    # Core top-level schema keys
    required_keys = ["decision", "summary", "primary_factor", "reasons", "policy_checks", "metrics_evaluated"]
    for k in required_keys:
        assert k in exp, f"Missing required explanation key: {k}"

    assert isinstance(exp["decision"], str)
    assert isinstance(exp["summary"], str)
    assert isinstance(exp["primary_factor"], str)
    assert isinstance(exp["reasons"], list)
    assert len(exp["reasons"]) > 0
    assert isinstance(exp["policy_checks"], dict)
    assert isinstance(exp["metrics_evaluated"], dict)

    # Policy checks schema
    expected_checks = [
        "fraud_risk", "instrument_status", "failure_streak",
        "recovery_viability", "value_threshold", "confidence_band", "velocity_risk"
    ]
    for check in expected_checks:
        assert check in exp["policy_checks"], f"Missing check: {check}"
        assert exp["policy_checks"][check] in ["PASSED", "FAILED", "ESCALATED", "UNKNOWN"]


def test_decision_consistency():
    """Test 2: For all representative transactions, decision_explanation['decision'] == policy_decision."""
    test_cases = [
        build_test_transaction(transaction_id="tc1", amount=2500.0, failure_reason="network_timeout", failure_category="transient"),
        build_test_transaction(transaction_id="tc2", amount=14500.0, failure_reason="network_timeout", failure_category="transient"),
        build_test_transaction(transaction_id="tc3", amount=7500.0, failure_reason="suspected_risk", failure_category="risk_related", ip_risk_score=0.92),
        build_test_transaction(transaction_id="tc4", amount=3200.0, failure_reason="bank_declined", failure_category="bank_decline", customer_historical_success_rate=0.0)
    ]

    for txn in test_cases:
        res = run_agent(txn)
        exp = res.get("decision_explanation")
        assert exp is not None
        assert exp["decision"] == res["policy_decision"], f"Mismatch for {txn['transaction_id']}: {exp['decision']} != {res['policy_decision']}"


def test_act_scenario():
    """Test 3: ACT Scenario -> checks accurately pass, reasons match approval, zero refusal flags."""
    txn = build_test_transaction(
        transaction_id="txn_act_test",
        amount=2500.0,
        failure_reason="network_timeout",
        failure_category="transient"
    )
    res = run_agent(txn)
    exp = res["decision_explanation"]

    assert res["policy_decision"] == "ACT"
    assert exp["decision"] == "ACT"
    assert exp["primary_factor"] == "STANDARD_POLICY_APPROVAL"
    
    # All safety checks must pass
    for check_name, status in exp["policy_checks"].items():
        assert status == "PASSED", f"Check {check_name} expected PASSED in ACT scenario, got {status}"

    # Reasons must state operational viability and zero violations
    reasons_text = " ".join(exp["reasons"])
    assert "meets or exceeds operational threshold" in reasons_text
    assert "Zero safety policy violations" in reasons_text
    assert "FAILED" not in reasons_text


def test_refuse_scenarios():
    """Test 4: REFUSE Scenarios -> fraud/risk, low probability, instrument failure, streak limit."""
    # 4a: Fraud / Risk Refusal
    txn_fraud = build_test_transaction(
        transaction_id="txn_fraud",
        amount=5000.0,
        failure_reason="suspected_risk",
        failure_category="risk_related",
        ip_risk_score=0.95
    )
    res_fraud = run_agent(txn_fraud)
    exp_fraud = res_fraud["decision_explanation"]

    assert res_fraud["policy_decision"] == "REFUSE"
    assert exp_fraud["decision"] == "REFUSE"
    assert exp_fraud["primary_factor"] == "HARD_SAFETY_FRAUD"
    assert exp_fraud["policy_checks"]["fraud_risk"] == "FAILED"
    assert any("fraud" in r.lower() or "risk" in r.lower() for r in exp_fraud["reasons"])

    # 4b: Low Recovery Probability Refusal
    txn_lowp = {"transaction_id": "t_low", "amount": 3200.0, "failure_reason": "bank_declined", "failure_category": "bank_decline"}
    res_lowp = evaluate_transaction_policy(txn_lowp, pred_prob=0.0514)
    exp_lowp = res_lowp["decision_explanation"]

    assert res_lowp["decision"] == "REFUSE"
    assert exp_lowp["decision"] == "REFUSE"
    assert exp_lowp["primary_factor"] == "PROBABILITY_BELOW_THRESHOLD"
    assert exp_lowp["policy_checks"]["recovery_viability"] == "FAILED"
    assert any("below operational viability threshold" in r for r in exp_lowp["reasons"])

    # 4c: Permanent Instrument Failure
    txn_inst = {"transaction_id": "t_inst", "amount": 1200.0, "failure_reason": "invalid_card", "failure_category": "payment_method_problem"}
    res_inst = evaluate_transaction_policy(txn_inst, pred_prob=0.85)
    exp_inst = res_inst["decision_explanation"]

    assert res_inst["decision"] == "REFUSE"
    assert exp_inst["decision"] == "REFUSE"
    assert exp_inst["primary_factor"] == "INSTRUMENT_FAILURE"
    assert exp_inst["policy_checks"]["instrument_status"] == "FAILED"
    assert any("instrument failure" in r.lower() for r in exp_inst["reasons"])

    # 4d: Consecutive Failure Streak Limit
    txn_strk = {"transaction_id": "t_strk", "amount": 1500.0, "failure_reason": "network_timeout", "consecutive_failure_streak": 4}
    res_strk = evaluate_transaction_policy(txn_strk, pred_prob=0.90)
    exp_strk = res_strk["decision_explanation"]

    assert res_strk["decision"] == "REFUSE"
    assert exp_strk["decision"] == "REFUSE"
    assert exp_strk["primary_factor"] == "STREAK_LIMIT_EXCEEDED"
    assert exp_strk["policy_checks"]["failure_streak"] == "FAILED"
    assert any("consecutive failure streak" in r.lower() for r in exp_strk["reasons"])


def test_escalate_scenario():
    """Test 5: ESCALATE Scenario -> high-value and uncertainty band checks escalate without false approval."""
    # 5a: High-Value Escalation
    txn_high = build_test_transaction(
        transaction_id="txn_high",
        amount=14500.0,
        failure_reason="network_timeout",
        failure_category="transient"
    )
    res_high = run_agent(txn_high)
    exp_high = res_high["decision_explanation"]

    assert res_high["policy_decision"] == "ESCALATE"
    assert exp_high["decision"] == "ESCALATE"
    assert exp_high["primary_factor"] == "HIGH_VALUE_THRESHOLD"
    assert exp_high["policy_checks"]["value_threshold"] == "ESCALATED"
    # Zero hard refusal checks failed
    assert exp_high["policy_checks"]["fraud_risk"] == "PASSED"
    assert exp_high["policy_checks"]["instrument_status"] == "PASSED"
    assert any("exceeds autonomous limit" in r for r in exp_high["reasons"])

    # 5b: Uncertainty Band Escalation
    txn_unc = {"transaction_id": "t_unc", "amount": 3000.0, "failure_reason": "network_timeout"}
    res_unc = evaluate_transaction_policy(txn_unc, pred_prob=0.35)
    exp_unc = res_unc["decision_explanation"]

    assert res_unc["decision"] == "ESCALATE"
    assert exp_unc["decision"] == "ESCALATE"
    assert exp_unc["primary_factor"] == "UNCERTAINTY_BAND"
    assert exp_unc["policy_checks"]["confidence_band"] == "ESCALATED"


def test_metric_traceability():
    """Test 6: Verify metrics in metrics_evaluated match actual transaction values and policy limits."""
    txn = build_test_transaction(
        transaction_id="txn_trace_test",
        amount=4250.75,
        failure_reason="network_timeout",
        failure_category="transient",
        ip_risk_score=0.12
    )
    res = run_agent(txn)
    exp = res["decision_explanation"]
    metrics = exp["metrics_evaluated"]

    assert metrics["transaction_amount"] == 4250.75
    assert metrics["high_value_limit"] == HIGH_VALUE_TRANSACTION_THRESHOLD
    assert metrics["streak_limit"] == 4
    assert metrics["ip_risk_limit"] == 0.70
    assert metrics["operational_threshold_tau"] == 0.35
    assert metrics["recovery_probability"] == round(float(res["recovery_probability"]), 4)
