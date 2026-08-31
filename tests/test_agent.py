import pytest
from agent.graph import run_agent, get_recovery_agent
from agent.nodes.policy import policy_guard
from agent.demo_data import build_test_transaction


def test_agent_graph_compiles():
    """Verify that the LangGraph workflow compiles cleanly."""
    agent = get_recovery_agent()
    assert agent is not None


def test_scenario_1_high_confidence_transient_failure():
    """SCENARIO 1: High-confidence transient failure -> Approved, Executed, Verified."""
    txn = build_test_transaction(
        transaction_id="txn_test_001",
        customer_id="cust_001",
        merchant_id="merch_01",
        amount=1500.0,
        currency="INR",
        payment_method="card",
        payment_network="visa",
        payment_channel="web",
        failure_reason="network_timeout",
        failure_category="transient",
        customer_age_days=100,
        customer_previous_transactions=10,
        customer_successful_transactions=9,
        customer_historical_success_rate=0.90,
        customer_lifetime_value=15000.0,
        customer_average_transaction=1500.0,
        customer_transaction_frequency_30d=3,
        is_subscription=1,
        subscription_age_days=90,
        is_first_transaction=0,
        previous_failures_24h=0,
        previous_failures_7d=0,
        previous_successes_30d=3,
        transactions_24h=1,
        transactions_7d=3,
        device_changed=0,
        location_changed=0,
        ip_risk_score=0.05,
        velocity_score=0.10,
        recovery_attempt_count=0,
        customer_contacted_today=0
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 1] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["recovery_probability"] > 0.0
    assert res["policy_decision"] == "ACT"
    assert res["agent_status"] in ["APPROVED", "COMPLETED"]
    assert res["selected_action"] is not None
    assert res["action_status"] == "executed"
    assert res["verification_status"] in ["SUCCESS", "FAILED", "PENDING"]


def test_scenario_2_retry_limit_exceeded():
    """SCENARIO 2: Retry limit exceeded -> Policy REFUSE."""
    txn = build_test_transaction(
        transaction_id="txn_test_002",
        customer_id="cust_002",
        merchant_id="merch_01",
        amount=2000.0,
        failure_reason="network_timeout",
        failure_category="transient",
        previous_failures_24h=4,
        previous_failures_7d=4,
        consecutive_failure_streak=4,
        recovery_attempt_count=4,
        customer_contacted_today=0
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 2] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["policy_decision"] == "REFUSE"
    assert any("streak" in v.lower() or "limit" in v.lower() for v in res["policy_violations"])
    assert res["action_status"] == "not_executed"


def test_scenario_3_high_value_payment():
    """SCENARIO 3: Transaction > ₹8,500 threshold -> Policy ESCALATE (AWAITING_APPROVAL)."""
    txn = build_test_transaction(
        transaction_id="txn_test_003",
        customer_id="cust_003",
        merchant_id="merch_01",
        amount=10000.0,  # Exceeds ₹8,500 threshold
        customer_average_transaction=10000.0,
        failure_reason="network_timeout",
        failure_category="transient",
        customer_historical_success_rate=0.95,
        previous_failures_24h=0
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 3] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["policy_decision"] == "ESCALATE"
    assert res["agent_status"] == "AWAITING_APPROVAL"
    assert res["action_status"] == "not_executed"


def test_scenario_4_risk_related_failure():
    """SCENARIO 4: Risk-related failure / High IP risk -> Policy REFUSE."""
    txn = build_test_transaction(
        transaction_id="txn_test_004",
        customer_id="cust_004",
        merchant_id="merch_01",
        amount=3000.0,
        failure_reason="suspected_risk",
        failure_category="risk_related",
        ip_risk_score=0.85,
        velocity_score=0.70,
        previous_failures_24h=0
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 4] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["policy_decision"] == "REFUSE"
    assert res["action_status"] == "not_executed"


def test_scenario_5_customer_contacted_today():
    """SCENARIO 5: Customer already contacted today / Permanent failure -> Policy REFUSE."""
    txn = build_test_transaction(
        transaction_id="txn_test_005",
        customer_id="cust_005",
        merchant_id="merch_01",
        amount=1000.0,
        failure_reason="invalid_card",
        failure_category="payment_method_problem",
        customer_contacted_today=1,  # Already contacted!
        previous_failures_24h=0
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 5] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["policy_decision"] == "REFUSE"
    assert res["action_status"] == "not_executed"


def test_scenario_6_low_recovery_probability():
    """SCENARIO 6: Recovery probability < 0.35 -> Policy REFUSE."""
    txn = build_test_transaction(
        transaction_id="txn_test_006",
        customer_id="cust_006",
        merchant_id="merch_01",
        amount=2000.0,
        failure_reason="invalid_card",
        failure_category="payment_method_problem",
        customer_historical_success_rate=0.0,
        customer_previous_transactions=1,
        customer_successful_transactions=0,
        previous_failures_24h=2,
        recovery_attempt_count=2,
        ip_risk_score=0.6,
        velocity_score=0.6
    )
    
    res = run_agent(txn)
    
    print(f"\n[Scenario 6] ML Prob: {res['recovery_probability']}, Rec: {res['recommended_action']}, Policy: {res['policy_decision']}")
    
    assert res["policy_decision"] == "REFUSE"
    assert res["action_status"] == "not_executed"



def test_scenario_7_policy_overrides_llm_recommendation():
    """SCENARIO 7: Policy overrides LLM recommendation. LLM says 'retry', policy BLOCKS it."""
    state = build_test_transaction(
        transaction_id="txn_test_007",
        amount=2000.0,
        failure_category="transient",
        failure_reason="network_timeout",
        previous_failures_24h=3,
        recovery_attempt_count=3,
        consecutive_failure_streak=4
    )

    
    state["recovery_probability"] = 0.90  # High ML score
    state["recommended_action"] = "retry" # Forced recommendation
    
    eval_state = policy_guard(state)
    
    print(f"\n[Scenario 7] Forced Rec: retry, Policy Decision: {eval_state['policy_decision']}")
    
    assert eval_state["policy_decision"] == "REFUSE"
    assert eval_state["agent_status"] == "BLOCKED"
    assert len(eval_state["policy_violations"]) > 0
    assert eval_state["selected_action"] is None
