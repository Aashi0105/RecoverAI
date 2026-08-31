"""
Unit and Concurrency Tests for Idempotency and Concurrency Safety (Upgrade #2).

Mocks only the external Razorpay API boundary.
Uses isolated in-memory or temporary SQLite database sessions to guarantee zero interference.
"""

import os
import sys
import threading
from unittest.mock import patch, MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.models import Base, PaymentExecutionClaim
from database.repository import (
    create_execution_claim,
    get_execution_claim,
    mark_execution_succeeded,
    mark_execution_failed_safe,
    mark_execution_unknown
)
from payment.executor import execute_recovery_policy


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """
    Creates an isolated SQLite test database engine for every test run.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    monkeypatch.setattr("payment.executor.engine", test_engine)
    monkeypatch.setattr("payment.executor.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("database.database.engine", test_engine)
    monkeypatch.setattr("database.database.SessionLocal", TestSessionLocal)

    yield TestSessionLocal


def test_sequential_duplicate_execution_returns_cached_result(setup_test_db):
    """
    Verifies that calling execute_recovery_policy twice sequentially for the same
    transaction_id invokes Razorpay API EXACTLY ONCE and returns cached results on 2nd call.
    """
    policy_eval = {
        "transaction_id": "txn_seq_001",
        "amount": 1500.0,
        "recovery_probability": 0.75,
        "decision": "ACT",
        "recommended_action": "payment_link"
    }

    mock_razorpay = MagicMock(return_value={
        "success": True,
        "payment_link_id": "plink_seq_001",
        "short_url": "https://rzp.io/i/seq001"
    })

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res1 = execute_recovery_policy(policy_eval, dry_run=False)
        assert res1["execution_status"] == "SUCCESS_CREATED"
        assert res1["external_api_called"] is True
        assert res1["razorpay_reference_id"] == "plink_seq_001"

        # 2nd Sequential Call
        res2 = execute_recovery_policy(policy_eval, dry_run=False)
        assert res2["execution_status"] == "IDEMPOTENT_SKIPPED"
        assert res2["external_api_called"] is False
        assert res2["razorpay_reference_id"] == "plink_seq_001"
        assert res2["short_url"] == "https://rzp.io/i/seq001"

        # Razorpay API called strictly once!
        assert mock_razorpay.call_count == 1


def test_concurrent_threads_execute_single_api_call(setup_test_db):
    """
    Verifies that multiple concurrent threads attempting execution for the same transaction_id
    race safely with a barrier, resulting in EXACTLY ONE Razorpay API call.
    """
    policy_eval = {
        "transaction_id": "txn_conc_001",
        "amount": 2500.0,
        "recovery_probability": 0.80,
        "decision": "ACT",
        "recommended_action": "payment_link"
    }

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results = []

    mock_razorpay = MagicMock(return_value={
        "success": True,
        "payment_link_id": "plink_conc_001",
        "short_url": "https://rzp.io/i/conc001"
    })

    def worker():
        barrier.wait()
        res = execute_recovery_policy(policy_eval, dry_run=False)
        results.append(res)

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(results) == num_threads
    assert mock_razorpay.call_count == 1

    success_count = sum(1 for r in results if r["execution_status"] == "SUCCESS_CREATED")
    skipped_count = sum(1 for r in results if r["execution_status"] in ["IDEMPOTENT_SKIPPED", "ALREADY_PROCESSING"])

    assert success_count == 1
    assert skipped_count == num_threads - 1


def test_atomic_database_duplicate_claim(setup_test_db):
    """
    Verifies that two independent database sessions attempting to insert the same idempotency key
    result in exactly one success and one safe IntegrityError rollback.
    """
    db1 = setup_test_db()
    db2 = setup_test_db()

    c1_ok, claim1 = create_execution_claim(db1, idempotency_key="idemp_db_001", payment_id="txn_db_001", action_type="payment_link", amount=1000.0)
    c2_ok, claim2 = create_execution_claim(db2, idempotency_key="idemp_db_001", payment_id="txn_db_001", action_type="payment_link", amount=1000.0)

    assert c1_ok is True
    assert claim1.status == "PROCESSING"

    assert c2_ok is False
    assert claim2.idempotency_key == "idemp_db_001"

    db1.close()
    db2.close()


def test_existing_succeeded_record_never_calls_external_api_again(setup_test_db):
    """
    Verifies that an existing SUCCEEDED record in the database blocks future external API calls.
    """
    db = setup_test_db()
    create_execution_claim(db, "idemp_txn_succ_001", "txn_succ_001", "payment_link", 5000.0)
    mark_execution_succeeded(db, "idemp_txn_succ_001", payment_link_id="plink_existing", short_url="https://rzp.io/i/exist")
    db.close()

    policy_eval = {"transaction_id": "txn_succ_001", "amount": 5000.0, "recovery_probability": 0.85, "decision": "ACT", "recommended_action": "payment_link"}
    mock_razorpay = MagicMock(return_value={"success": True, "payment_link_id": "plink_existing", "short_url": "https://rzp.io/i/exist"})

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res = execute_recovery_policy(policy_eval, dry_run=False)
        assert res["execution_status"] == "IDEMPOTENT_SKIPPED"
        assert res["external_api_called"] is False
        assert mock_razorpay.call_count == 0


def test_processing_record_blocks_duplicate_external_execution(setup_test_db):
    """
    Verifies that an existing PROCESSING claim in the database blocks concurrent/re-try execution.
    """
    db = setup_test_db()
    create_execution_claim(db, "idemp_txn_proc_001", "txn_proc_001", "payment_link", 3000.0)
    db.close()

    policy_eval = {"transaction_id": "txn_proc_001", "amount": 3000.0, "recovery_probability": 0.85, "decision": "ACT", "recommended_action": "payment_link"}
    mock_razorpay = MagicMock(return_value={"success": True, "payment_link_id": "plink_proc", "short_url": "https://rzp.io/i/proc"})

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res = execute_recovery_policy(policy_eval, dry_run=False)
        assert res["execution_status"] == "ALREADY_PROCESSING"
        assert res["external_api_called"] is False
        assert mock_razorpay.call_count == 0


def test_unknown_external_result_blocks_automatic_retry(setup_test_db):
    """
    Verifies that a transaction in UNKNOWN_EXTERNAL_RESULT state prohibits automatic re-execution.
    """
    db = setup_test_db()
    create_execution_claim(db, "idemp_txn_unk_001", "txn_unk_001", "payment_link", 4000.0)
    mark_execution_unknown(db, "idemp_txn_unk_001", "Simulated timeout during API call")
    db.close()

    policy_eval = {"transaction_id": "txn_unk_001", "amount": 4000.0, "recovery_probability": 0.85, "decision": "ACT", "recommended_action": "payment_link"}
    mock_razorpay = MagicMock(return_value={"success": True, "payment_link_id": "plink_unk", "short_url": "https://rzp.io/i/unk"})

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res = execute_recovery_policy(policy_eval, dry_run=False)
        assert res["execution_status"] == "UNKNOWN_EXTERNAL_RESULT"
        assert res["external_api_called"] is False
        assert mock_razorpay.call_count == 0




def test_safe_local_failure_is_recorded_correctly(setup_test_db):
    """
    Verifies that local failures before API call (e.g. missing credentials) are marked FAILED_SAFE.
    """
    policy_eval = {"transaction_id": "txn_safe_fail", "amount": 1200.0, "recovery_probability": 0.70, "decision": "ACT", "recommended_action": "payment_link"}

    with patch("payment.executor.is_razorpay_configured", return_value=False):
        res = execute_recovery_policy(policy_eval, dry_run=False)
        assert res["execution_status"] == "CREDENTIALS_MISSING"
        assert res["external_api_called"] is False

    db = setup_test_db()
    claim = get_execution_claim(db, "idemp_txn_safe_fail")
    assert claim is not None
    assert claim.status == "FAILED_SAFE"
    db.close()


def test_missing_transaction_id_fails_safely(setup_test_db):
    """
    Verifies missing or default transaction_id triggers policy/execution default safety.
    """
    policy_eval = {"amount": 500.0, "recovery_probability": 0.60, "decision": "REFUSE", "recommended_action": "no_action"}
    mock_razorpay = MagicMock()

    with patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):
        res = execute_recovery_policy(policy_eval, dry_run=False)
        assert res["execution_status"] == "BLOCKED_BY_POLICY"
        assert res["external_api_called"] is False
        assert mock_razorpay.call_count == 0


def test_different_transaction_ids_execute_independently(setup_test_db):
    """
    Verifies that distinct transaction IDs execute independently without cross-blocking.
    """
    p1 = {"transaction_id": "txn_indep_A", "amount": 1000.0, "recovery_probability": 0.75, "decision": "ACT", "recommended_action": "payment_link"}
    p2 = {"transaction_id": "txn_indep_B", "amount": 2000.0, "recovery_probability": 0.85, "decision": "ACT", "recommended_action": "payment_link"}

    mock_razorpay = MagicMock(side_effect=[
        {"success": True, "payment_link_id": "plink_A", "short_url": "https://rzp.io/i/A"},
        {"success": True, "payment_link_id": "plink_B", "short_url": "https://rzp.io/i/B"}
    ])

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res1 = execute_recovery_policy(p1, dry_run=False)
        res2 = execute_recovery_policy(p2, dry_run=False)

        assert res1["execution_status"] == "SUCCESS_CREATED"
        assert res2["execution_status"] == "SUCCESS_CREATED"
        assert mock_razorpay.call_count == 2


# -----------------------------------------------------------------------------
# ADVERSARIAL TEST SUITE
# -----------------------------------------------------------------------------

def test_adversarial_same_txn_different_amount(setup_test_db):
    """
    Adversarial A: Same transaction ID with different amount on 2nd invocation.
    Guarantees canonical transaction identity blocks duplicate external payment link creation.
    """
    p1 = {"transaction_id": "txn_adv_001", "amount": 1000.0, "recovery_probability": 0.80, "decision": "ACT", "recommended_action": "payment_link"}
    p2 = {"transaction_id": "txn_adv_001", "amount": 9999.0, "recovery_probability": 0.80, "decision": "ACT", "recommended_action": "payment_link"}

    mock_razorpay = MagicMock(return_value={"success": True, "payment_link_id": "plink_adv1", "short_url": "https://rzp.io/i/adv1"})

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res1 = execute_recovery_policy(p1, dry_run=False)
        res2 = execute_recovery_policy(p2, dry_run=False)

        assert res1["execution_status"] == "SUCCESS_CREATED"
        assert res2["execution_status"] == "IDEMPOTENT_SKIPPED"
        assert mock_razorpay.call_count == 1


def test_adversarial_same_txn_different_customer_info(setup_test_db):
    """
    Adversarial B: Same transaction ID with modified customer info.
    Guarantees cannot bypass idempotency claim to create another payment link.
    """
    p1 = {"transaction_id": "txn_adv_cust", "amount": 1000.0, "recovery_probability": 0.80, "decision": "ACT", "recommended_action": "payment_link"}
    cust1 = {"name": "Alice", "email": "alice@example.com"}
    cust2 = {"name": "Hacker Bob", "email": "bob@example.com"}

    mock_razorpay = MagicMock(return_value={"success": True, "payment_link_id": "plink_cust", "short_url": "https://rzp.io/i/cust"})

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):

        res1 = execute_recovery_policy(p1, customer_info=cust1, dry_run=False)
        res2 = execute_recovery_policy(p1, customer_info=cust2, dry_run=False)

        assert res1["execution_status"] == "SUCCESS_CREATED"
        assert res2["execution_status"] == "IDEMPOTENT_SKIPPED"
        assert mock_razorpay.call_count == 1


def test_adversarial_unknown_decision_payload(setup_test_db):
    """
    Adversarial C: Non-ACT decision payload bypass attempt.
    """
    p_bad = {"transaction_id": "txn_bad_dec", "amount": 1000.0, "recovery_probability": 0.80, "decision": "HACKED_APPROVAL", "recommended_action": "payment_link"}
    mock_razorpay = MagicMock()

    with patch("payment.executor.create_razorpay_test_payment_link", mock_razorpay):
        res = execute_recovery_policy(p_bad, dry_run=False)
        assert res["execution_status"] == "BLOCKED_BY_POLICY"
        assert mock_razorpay.call_count == 0


def test_adversarial_exception_during_api_execution(setup_test_db):
    """
    Adversarial D: Exception thrown during Razorpay execution transitions state to UNKNOWN_EXTERNAL_RESULT.
    """
    p_exc = {"transaction_id": "txn_exc_001", "amount": 1000.0, "recovery_probability": 0.80, "decision": "ACT", "recommended_action": "payment_link"}

    with patch("payment.executor.is_razorpay_configured", return_value=True), \
         patch("payment.executor.create_razorpay_test_payment_link", side_effect=RuntimeError("Network Timeout")):

        res = execute_recovery_policy(p_exc, dry_run=False)
        assert res["execution_status"] == "UNKNOWN_EXTERNAL_RESULT"

    db = setup_test_db()
    claim = get_execution_claim(db, "idemp_txn_exc_001")
    assert claim is not None
    assert claim.status == "UNKNOWN_EXTERNAL_RESULT"
    db.close()
