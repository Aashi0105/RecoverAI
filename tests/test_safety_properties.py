"""
Comprehensive Property-Based & Boundary Safety Test Suite for RecoverAI.

Tests:
1. Hard Safety Refusal Invariants (Risk, Instrument Failure, Failure Streak, Low Probability).
2. Escalation Boundaries (High Value Amount, Uncertainty Band, Risk Warnings).
3. Precedence Order (Hard Refusal > Escalation > ACT).
4. Execution Safety Guarantee (REFUSE / ESCALATE -> external_api_called == False).
5. Robust randomized & boundary property testing (10,000+ generated test combinations).
"""

import pytest
import numpy as np

from agent.nodes.policy import (
    evaluate_transaction_policy,
    HIGH_VALUE_TRANSACTION_THRESHOLD,
    UNCERTAINTY_BAND_LOW,
    UNCERTAINTY_BAND_HIGH
)
from payment.executor import execute_recovery_policy

# Try importing Hypothesis for property-based generation
try:
    from hypothesis import given, strategies as st, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

ALL_FAILURE_REASONS = [
    "network_timeout", "technical_error", "insufficient_funds",
    "authentication_failed", "limit_exceeded", "bank_declined",
    "customer_cancelled", "suspected_risk", "invalid_card", "card_expired"
]

ALL_PAYMENT_METHODS = ["upi", "credit_card", "debit_card", "net_banking", "wallet"]

# -----------------------------------------------------------------------------
# 1. INVARIANT A: SUSPECTED RISK / FRAUD MUST ALWAYS REFUSE
# -----------------------------------------------------------------------------
def test_invariant_a_suspected_risk_always_refuses():
    """
    INVARIANT A: If failure_reason == 'suspected_risk' OR category == 'risk_related' OR ip_risk > 0.70,
    decision MUST be REFUSE, regardless of probability or amount.
    """
    for prob in [0.0, 0.35, 0.50, 0.85, 0.99, 1.0]:
        for amt in [0.0, 100.0, 5000.0, 8500.0, 50000.0]:
            for streak in [0, 1, 3, 5]:
                txn = {
                    "transaction_id": "txn_test_risk",
                    "amount": amt,
                    "failure_reason": "suspected_risk",
                    "consecutive_failure_streak": streak
                }
                res = evaluate_transaction_policy(txn, pred_prob=prob)
                assert res["decision"] == "REFUSE", f"Failed for prob={prob}, amt={amt}: got {res['decision']}"
                assert res["recommended_action"] == "no_action"
                assert any("HARD_SAFETY_REFUSE" in r for r in res["triggered_rules"])


def test_invariant_a_high_ip_risk_always_refuses():
    for prob in [0.0, 0.50, 0.99]:
        txn = {
            "transaction_id": "txn_test_ip_risk",
            "amount": 1000.0,
            "failure_reason": "network_timeout",
            "ip_risk_score": 0.71
        }
        res = evaluate_transaction_policy(txn, pred_prob=prob)
        assert res["decision"] == "REFUSE"
        assert res["recommended_action"] == "no_action"


# -----------------------------------------------------------------------------
# 2. INVARIANT B: PERMANENT INSTRUMENT FAILURE MUST ALWAYS REFUSE
# -----------------------------------------------------------------------------
def test_invariant_b_permanent_card_failure_always_refuses():
    """
    INVARIANT B: If failure_reason in ['invalid_card', 'card_expired'] OR category == 'payment_method_problem',
    decision MUST be REFUSE.
    """
    for reason in ["invalid_card", "card_expired"]:
        for prob in [0.0, 0.50, 0.99]:
            for amt in [100.0, 10000.0]:
                txn = {
                    "transaction_id": "txn_test_card",
                    "amount": amt,
                    "failure_reason": reason
                }
                res = evaluate_transaction_policy(txn, pred_prob=prob)
                assert res["decision"] == "REFUSE"
                assert res["recommended_action"] == "no_action"


# -----------------------------------------------------------------------------
# 3. INVARIANT C: FAILURE STREAK LIMIT (>= 4) MUST ALWAYS REFUSE
# -----------------------------------------------------------------------------
def test_invariant_c_streak_limit_always_refuses():
    """
    INVARIANT C: If consecutive_failure_streak >= 4, decision MUST be REFUSE.
    """
    for streak in [4, 5, 10, 100]:
        for prob in [0.35, 0.75, 0.99]:
            txn = {
                "transaction_id": "txn_test_streak",
                "amount": 1000.0,
                "failure_reason": "network_timeout",
                "consecutive_failure_streak": streak
            }
            res = evaluate_transaction_policy(txn, pred_prob=prob)
            assert res["decision"] == "REFUSE"
            assert res["recommended_action"] == "no_action"


# -----------------------------------------------------------------------------
# 4. INVARIANT D: LOW PROBABILITY (< 0.35) MUST ALWAYS REFUSE
# -----------------------------------------------------------------------------
def test_invariant_d_low_probability_always_refuses():
    """
    INVARIANT D: If recovery probability < 0.35, decision MUST be REFUSE.
    """
    for prob in [0.0, 0.10, 0.25, 0.3499]:
        txn = {
            "transaction_id": "txn_test_low_p",
            "amount": 1000.0,
            "failure_reason": "network_timeout",
            "consecutive_failure_streak": 0
        }
        res = evaluate_transaction_policy(txn, pred_prob=prob)
        assert res["decision"] == "REFUSE"
        assert res["recommended_action"] == "no_action"


# -----------------------------------------------------------------------------
# 5. INVARIANT E & F & G: ESCALATION BOUNDARIES
# -----------------------------------------------------------------------------
def test_invariant_e_high_value_escalation():
    """
    INVARIANT E: If amount >= 8500.00 and no refusal rule triggered, decision MUST be ESCALATE.
    """
    for amt in [8500.00, 8500.01, 15000.00, 100000.00]:
        txn = {
            "transaction_id": "txn_test_high_val",
            "amount": amt,
            "failure_reason": "network_timeout",
            "consecutive_failure_streak": 0
        }
        res = evaluate_transaction_policy(txn, pred_prob=0.75)
        assert res["decision"] == "ESCALATE"
        assert any("HIGH_VALUE_ESCALATE" in r for r in res["triggered_rules"])


def test_invariant_f_uncertainty_band_escalation():
    """
    INVARIANT F: If 0.35 <= prob <= 0.38 and no refusal rule triggered, decision MUST be ESCALATE.
    """
    for prob in [0.35, 0.36, 0.38]:
        txn = {
            "transaction_id": "txn_test_uncertain",
            "amount": 1000.0,
            "failure_reason": "network_timeout",
            "consecutive_failure_streak": 0
        }
        res = evaluate_transaction_policy(txn, pred_prob=prob)
        assert res["decision"] == "ESCALATE"
        assert any("BOUNDARY_UNCERTAINTY_ESCALATE" in r for r in res["triggered_rules"])



def test_invariant_g_risk_warning_escalation():
    """
    INVARIANT G: If velocity_score > 0.65 AND ip_risk_score > 0.50, decision MUST be ESCALATE.
    """
    txn = {
        "transaction_id": "txn_test_warning",
        "amount": 1000.0,
        "failure_reason": "network_timeout",
        "velocity_score": 0.66,
        "ip_risk_score": 0.51,
        "consecutive_failure_streak": 0
    }
    res = evaluate_transaction_policy(txn, pred_prob=0.75)
    assert res["decision"] == "ESCALATE"
    assert any("RISK_WARNING_ESCALATE" in r for r in res["triggered_rules"])


# -----------------------------------------------------------------------------
# 6. STAGE PRECEDENCE TESTING (Hard Refusal > Escalation > ACT)
# -----------------------------------------------------------------------------
def test_precedence_hard_refusal_overrides_escalation():
    """
    Verifies Hard Refusal (Stage 1) takes absolute precedence over Escalation (Stage 2).
    Scenario: High value transaction (amount=10,000) WITH suspected_risk.
    Result: MUST be REFUSE, NOT ESCALATE.
    """
    txn = {
        "transaction_id": "txn_prec_1",
        "amount": 10000.00,
        "failure_reason": "suspected_risk",
        "consecutive_failure_streak": 0
    }
    res = evaluate_transaction_policy(txn, pred_prob=0.95)
    assert res["decision"] == "REFUSE"
    assert res["recommended_action"] == "no_action"


def test_precedence_streak_limit_overrides_high_probability():
    """
    Verifies streak limit takes precedence over high probability.
    """
    txn = {
        "transaction_id": "txn_prec_2",
        "amount": 500.0,
        "failure_reason": "network_timeout",
        "consecutive_failure_streak": 4
    }
    res = evaluate_transaction_policy(txn, pred_prob=0.99)
    assert res["decision"] == "REFUSE"


# -----------------------------------------------------------------------------
# 7. EXECUTION SAFETY GUARANTEE CHECK
# -----------------------------------------------------------------------------
def test_execution_safety_guarantee():
    """
    Proves that payment/executor.py NEVER makes external API calls for REFUSE or ESCALATE.
    """
    for dec in ["REFUSE", "ESCALATE"]:
        policy_eval = {
            "transaction_id": "txn_exec_safety",
            "amount": 1000.0,
            "decision": dec,
            "recommended_action": "no_action" if dec == "REFUSE" else "payment_link",
            "action_cost": 0.0
        }
        exec_res = execute_recovery_policy(policy_eval, dry_run=False) # Test even with dry_run=False
        assert exec_res["external_api_called"] is False
        assert exec_res["execution_status"] in ["BLOCKED_BY_POLICY", "REFUSED_BY_POLICY", "ESCALATED_TO_MERCHANT"]



# -----------------------------------------------------------------------------
# 8. EXHAUSTIVE RANDOMIZED & BOUNDARY PROPERTY GENERATOR (10,000+ CASES)
# -----------------------------------------------------------------------------
def test_randomized_boundary_property_suite():
    """
    Generates 10,000 randomized transaction configurations across boundary points
    and asserts universal safety invariants hold across 100% of cases.
    """
    np.random.seed(42)
    probs = [0.0, 0.10, 0.3199, 0.32, 0.3499, 0.35, 0.3501, 0.38, 0.3801, 0.50, 0.75, 0.99, 1.0]
    amounts = [0.0, 0.01, 100.0, 8499.99, 8500.00, 8500.01, 15000.0, 100000.0]
    streaks = [0, 1, 2, 3, 4, 5, 10]
    reasons = ALL_FAILURE_REASONS
    
    total_runs = 0
    
    for _ in range(10000):
        prob = float(np.random.choice(probs))
        amt = float(np.random.choice(amounts))
        streak = int(np.random.choice(streaks))
        reason = str(np.random.choice(reasons))
        ip_risk = float(np.random.choice([0.0, 0.40, 0.51, 0.71, 0.90]))
        vel = float(np.random.choice([0.0, 0.50, 0.66, 0.90]))
        
        txn = {
            "transaction_id": f"txn_prop_{total_runs}",
            "amount": amt,
            "failure_reason": reason,
            "consecutive_failure_streak": streak,
            "ip_risk_score": ip_risk,
            "velocity_score": vel
        }
        
        res = evaluate_transaction_policy(txn, pred_prob=prob)
        dec = res["decision"]
        
        # UNIVERSAL INVARIANT CHECKS
        is_refuse_condition = (
            reason in ["suspected_risk", "invalid_card", "card_expired"] or
            ip_risk > 0.70 or
            streak >= 4 or
            prob < 0.35
        )
        
        if is_refuse_condition:
            assert dec == "REFUSE", f"Violation at run {total_runs}: expected REFUSE, got {dec} for txn {txn}, prob={prob}"
            assert res["recommended_action"] == "no_action"
            
        if dec == "ACT":
            assert prob >= 0.35, "ACT decision allowed P < 0.35!"
            assert amt < 8500.00, "ACT decision allowed amount >= 8500.00!"
            assert streak < 4, "ACT decision allowed streak >= 4!"
            assert reason not in ["suspected_risk", "invalid_card", "card_expired"], f"ACT decision allowed risky reason {reason}!"
            assert ip_risk <= 0.70, "ACT decision allowed ip_risk > 0.70!"
            assert not (UNCERTAINTY_BAND_LOW <= prob <= UNCERTAINTY_BAND_HIGH), "ACT decision allowed probability in uncertainty band!"
            
        total_runs += 1

    print(f"\n ✅ EXHAUSTIVE PROPERTY TEST PASSED: {total_runs:,} randomized transaction cases verified cleanly.")


# -----------------------------------------------------------------------------
# 9. HYPOTHESIS PROPERTY TESTS (IF HYPOTHESIS IS AVAILABLE)
# -----------------------------------------------------------------------------
if HAS_HYPOTHESIS:
    @settings(max_examples=500)
    @given(
        prob=st.floats(min_value=0.0, max_value=1.0),
        amt=st.floats(min_value=0.0, max_value=100000.0),
        streak=st.integers(min_value=0, max_value=20),
        reason=st.sampled_from(ALL_FAILURE_REASONS)
    )
    def test_hypothesis_safety_invariants(prob, amt, streak, reason):
        txn = {
            "transaction_id": "txn_hypo",
            "amount": amt,
            "failure_reason": reason,
            "consecutive_failure_streak": streak
        }
        res = evaluate_transaction_policy(txn, pred_prob=prob)
        dec = res["decision"]
        
        if reason in ["suspected_risk", "invalid_card", "card_expired"] or streak >= 4 or prob < 0.35:
            assert dec == "REFUSE"
        elif amt >= 8500.00 or (UNCERTAINTY_BAND_LOW <= prob <= UNCERTAINTY_BAND_HIGH):
            assert dec in ["ESCALATE", "REFUSE"]
        if dec == "ACT":
            assert prob >= 0.35 and amt < 8500.00 and streak < 4
