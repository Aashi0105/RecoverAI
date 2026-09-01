"""
Comprehensive Test Suite for Phase 3B: Human-in-the-Loop (HITL) Merchant Approval Flow.

Tests:
1. Escalated transaction enters awaiting approval state and auto-populates ApprovalRequest.
2. GET /api/v1/approvals/pending returns the pending transaction.
3. GET /api/v1/approvals/{id} returns full decision, ML scores, and diagnostic context.
4. POST /api/v1/approvals/{id}/approve successfully approves and executes action.
5. Duplicate approval is idempotent and does NOT duplicate external payment action.
6. POST /api/v1/approvals/{id}/reject safely rejects and ensures zero payments are executed.
7. Unknown transaction returns HTTP 404 for GET, approve, and reject.
8. CRITICAL FINTECH INVARIANT: Blocked fraud/risk transaction CANNOT be approved by human.
9. Approve after reject returns HTTP 409 Conflict.
10. Reject after approve returns HTTP 409 Conflict.
11. Concurrent approval attempts result in exactly ONE execution (thread barrier test).
12. Audit trail immutably records human approval (actor=MERCHANT_HUMAN).
13. Audit trail immutably records human rejection (actor=MERCHANT_HUMAN).
14. Standard automatic approved flow (P >= 0.35, amount < 8500) executes without approval queue.
"""

import threading
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.database import Base, get_db
from database.models import FailedPayment, RecoveryAction, AuditLog, PaymentExecutionClaim, ApprovalRequest
from database.repository import save_recovery_audit, get_approval_request
from agent.graph import run_agent
from agent.demo_data import build_test_transaction
from backend.main import app


# -----------------------------------------------------------------------------
# Test Database Fixture (In-Memory SQLite with StaticPool)
# -----------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_session():
    """Creates a fresh in-memory SQLite database for test isolation with StaticPool."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def seed_escalated_transaction(db_session, txn_id="txn_hitl_001", amount=12000.0):
    """Helper to run an escalated transaction through agent and save to DB."""
    txn = build_test_transaction(
        transaction_id=txn_id,
        customer_id="cust_high_value",
        merchant_id="merch_enterprise",
        amount=amount,  # > ₹8,500 threshold -> triggers ESCALATE
        customer_average_transaction=amount,
        failure_reason="network_timeout",
        failure_category="transient",
        customer_historical_success_rate=0.95,
        previous_failures_24h=0,
        recovery_attempt_count=0,
        ip_risk_score=0.05,
        velocity_score=0.10
    )
    result = run_agent(txn)
    save_recovery_audit(db_session, result)
    return result


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_1_escalated_transaction_enters_awaiting_approval_queue(db_session):
    """Verifies that an escalated transaction creates an ApprovalRequest in PENDING_APPROVAL status."""
    result = seed_escalated_transaction(db_session, txn_id="txn_queue_001", amount=15000.0)

    assert result["policy_decision"] == "ESCALATE"
    assert result["agent_status"] == "AWAITING_APPROVAL"
    assert result["action_status"] == "not_executed"

    approval_req = get_approval_request(db_session, "txn_queue_001")
    assert approval_req is not None
    assert approval_req.transaction_id == "txn_queue_001"
    assert approval_req.amount == 15000.0
    assert approval_req.status == "PENDING_APPROVAL"
    assert approval_req.recommended_action in ["retry", "payment_link"]
    assert approval_req.human_decision is None


def test_2_pending_approvals_endpoint(client, db_session):
    """Verifies GET /api/v1/approvals/pending returns all pending transactions."""
    seed_escalated_transaction(db_session, txn_id="txn_pend_001", amount=9000.0)
    seed_escalated_transaction(db_session, txn_id="txn_pend_002", amount=14000.0)

    response = client.get("/api/v1/approvals/pending")
    assert response.status_code == 200

    data = response.json()
    assert data["total_pending"] >= 2
    txn_ids = [item["transaction_id"] for item in data["approvals"]]
    assert "txn_pend_001" in txn_ids
    assert "txn_pend_002" in txn_ids


def test_3_get_approval_details_endpoint(client, db_session):
    """Verifies GET /api/v1/approvals/{id} returns rich diagnostic & policy metadata."""
    seed_escalated_transaction(db_session, txn_id="txn_detail_001", amount=11500.0)

    response = client.get("/api/v1/approvals/txn_detail_001")
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "txn_detail_001"
    assert data["amount"] == 11500.0
    assert data["status"] == "PENDING_APPROVAL"
    assert data["policy_decision"] == "ESCALATE"
    assert data["recommended_action"] in ["retry", "payment_link"]
    assert data["diagnosis_summary"] is not None


def test_4_approve_endpoint_executes_action_safely(client, db_session):
    """Verifies POST /api/v1/approvals/{id}/approve approves and executes recovery action."""
    seed_escalated_transaction(db_session, txn_id="txn_appr_001", amount=10000.0)

    payload = {"notes": "Approved by senior finance manager.", "dry_run": True}
    response = client.post("/api/v1/approvals/txn_appr_001/approve", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "txn_appr_001"
    assert data["status"] == "EXECUTED"
    assert data["human_decision"] == "APPROVED"
    assert data["human_notes"] == "Approved by senior finance manager."
    assert data["execution_details"] is not None

    # Verify DB state
    req = get_approval_request(db_session, "txn_appr_001")
    assert req.status == "EXECUTED"
    assert req.human_decision == "APPROVED"
    assert req.resolved_at is not None


def test_5_duplicate_approval_is_idempotent(client, db_session):
    """Verifies that duplicate approval calls do NOT trigger multiple payments."""
    seed_escalated_transaction(db_session, txn_id="txn_dup_001", amount=9500.0)

    # First approval
    resp1 = client.post("/api/v1/approvals/txn_dup_001/approve", json={"notes": "First approval", "dry_run": True})
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "EXECUTED"

    # Second approval (Idempotent call)
    resp2 = client.post("/api/v1/approvals/txn_dup_001/approve", json={"notes": "Repeated approval", "dry_run": True})
    assert resp2.status_code == 200
    assert "already been approved" in resp2.json()["message"].lower()

    # RecoveryAction count must be exactly 1
    actions = db_session.query(RecoveryAction).filter(RecoveryAction.payment_id == "txn_dup_001").all()
    assert len(actions) == 1


def test_6_reject_endpoint_ensures_zero_execution(client, db_session):
    """Verifies POST /api/v1/approvals/{id}/reject sets REJECTED_BY_HUMAN and zero payments execute."""
    seed_escalated_transaction(db_session, txn_id="txn_rej_001", amount=13000.0)

    payload = {"notes": "Customer requested account pause."}
    response = client.post("/api/v1/approvals/txn_rej_001/reject", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "txn_rej_001"
    assert data["status"] == "REJECTED_BY_HUMAN"
    assert data["human_decision"] == "REJECTED"

    # Verify zero recovery actions executed
    actions = db_session.query(RecoveryAction).filter(RecoveryAction.payment_id == "txn_rej_001").all()
    assert len(actions) == 0

    # Verify DB state
    req = get_approval_request(db_session, "txn_rej_001")
    assert req.status == "REJECTED_BY_HUMAN"
    assert req.human_decision == "REJECTED"


def test_7_unknown_transaction_returns_404(client):
    """Verifies that non-existent transaction IDs return HTTP 404."""
    resp_get = client.get("/api/v1/approvals/txn_nonexistent_999")
    assert resp_get.status_code == 404

    resp_appr = client.post("/api/v1/approvals/txn_nonexistent_999/approve", json={})
    assert resp_appr.status_code == 404

    resp_rej = client.post("/api/v1/approvals/txn_nonexistent_999/reject", json={})
    assert resp_rej.status_code == 404


def test_8_blocked_fraud_transaction_cannot_be_approved_by_human(client, db_session):
    """
    CRITICAL FINTECH INVARIANT:
    A transaction blocked by deterministic safety policy (e.g. fraud/IP risk)
    CANNOT be approved or executed by human intervention.
    """
    # Seed a high-risk transaction that policy blocked
    blocked_txn = build_test_transaction(
        transaction_id="txn_fraud_blocked_001",
        amount=5000.0,
        failure_reason="suspected_risk",
        failure_category="risk_related",
        ip_risk_score=0.92  # High fraud risk -> REFUSE
    )
    result = run_agent(blocked_txn)
    assert result["policy_decision"] == "REFUSE"
    save_recovery_audit(db_session, result)

    # Attempt human approval on blocked transaction
    response = client.post("/api/v1/approvals/txn_fraud_blocked_001/approve", json={"notes": "Attempt override"})
    assert response.status_code == 409  # Conflict!
    data = response.json()
    assert "CANNOT_APPROVE_BLOCKED" in data["detail"]["error"] or "blocked" in data["detail"]["message"].lower()

    # Zero actions executed
    actions = db_session.query(RecoveryAction).filter(RecoveryAction.payment_id == "txn_fraud_blocked_001").all()
    assert len(actions) == 0


def test_9_approve_after_reject_returns_conflict(client, db_session):
    """Verifies that an already-rejected transaction cannot subsequently be approved."""
    seed_escalated_transaction(db_session, txn_id="txn_rej_first_001", amount=10500.0)

    # Reject first
    client.post("/api/v1/approvals/txn_rej_first_001/reject", json={"notes": "Rejected"})

    # Try approve
    response = client.post("/api/v1/approvals/txn_rej_first_001/approve", json={"notes": "Late approve"})
    assert response.status_code == 409
    assert "already been rejected" in response.json()["detail"]["message"].lower()


def test_10_reject_after_approve_returns_conflict(client, db_session):
    """Verifies that an already-approved and executed transaction cannot subsequently be rejected."""
    seed_escalated_transaction(db_session, txn_id="txn_appr_first_001", amount=10500.0)

    # Approve first
    client.post("/api/v1/approvals/txn_appr_first_001/approve", json={"notes": "Approved", "dry_run": True})

    # Try reject
    response = client.post("/api/v1/approvals/txn_appr_first_001/reject", json={"notes": "Late reject"})
    assert response.status_code == 409
    assert "already been approved" in response.json()["detail"]["message"].lower()


def test_11_concurrent_approval_requests_execute_exactly_once(client, db_session):
    """
    CONCURRENCY TEST:
    Spawns 5 concurrent threads attempting to approve the same pending transaction via API.
    Idempotency claim and thread locking guarantee that exactly ONE execution occurs.
    """
    seed_escalated_transaction(db_session, txn_id="txn_concur_hitl_001", amount=16000.0)

    num_threads = 5
    barrier = threading.Barrier(num_threads)
    results = []

    def worker():
        barrier.wait()
        res = client.post(
            "/api/v1/approvals/txn_concur_hitl_001/approve",
            json={"notes": "Concurrent approval attempt", "dry_run": True}
        )
        results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should report success (200 OK)
    assert len(results) == num_threads
    for resp in results:
        assert resp.status_code == 200

    # But exactly 1 RecoveryAction must be created
    actions = db_session.query(RecoveryAction).filter(RecoveryAction.payment_id == "txn_concur_hitl_001").all()
    assert len(actions) == 1


def test_12_audit_trail_records_human_approval(client, db_session):
    """Verifies that an immutable AuditLog entry is recorded with actor=MERCHANT_HUMAN on approval."""
    seed_escalated_transaction(db_session, txn_id="txn_audit_appr_001", amount=11000.0)

    client.post("/api/v1/approvals/txn_audit_appr_001/approve", json={"notes": "Audit verification note", "dry_run": True})

    human_audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.payment_id == "txn_audit_appr_001", AuditLog.actor == "MERCHANT_HUMAN")
        .first()
    )
    assert human_audit is not None
    assert human_audit.event_type == "MERCHANT_APPROVAL"
    assert human_audit.details["decision"] == "APPROVED"
    assert human_audit.details["notes"] == "Audit verification note"


def test_13_audit_trail_records_human_rejection(client, db_session):
    """Verifies that an immutable AuditLog entry is recorded with actor=MERCHANT_HUMAN on rejection."""
    seed_escalated_transaction(db_session, txn_id="txn_audit_rej_001", amount=11000.0)

    client.post("/api/v1/approvals/txn_audit_rej_001/reject", json={"notes": "Reject audit note"})

    human_audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.payment_id == "txn_audit_rej_001", AuditLog.actor == "MERCHANT_HUMAN")
        .first()
    )
    assert human_audit is not None
    assert human_audit.event_type == "MERCHANT_REJECTION"
    assert human_audit.details["decision"] == "REJECTED"
    assert human_audit.details["notes"] == "Reject audit note"


def test_14_standard_approved_flow_bypasses_approval_queue(db_session):
    """
    Verifies that a standard eligible transaction (amount < ₹8,500, transient, P >= 0.35)
    is ACT/APPROVED automatically and does NOT sit in PENDING_APPROVAL.
    """
    txn = build_test_transaction(
        transaction_id="txn_auto_001",
        amount=2500.0,  # Below ₹8,500
        failure_reason="network_timeout",
        failure_category="transient",
        ip_risk_score=0.05
    )
    res = run_agent(txn)
    assert res["policy_decision"] == "ACT"
    assert res["agent_status"] in ["APPROVED", "COMPLETED"]

    save_recovery_audit(db_session, res)

    # Should NOT be in ApprovalRequest queue
    req = get_approval_request(db_session, "txn_auto_001")
    assert req is None
