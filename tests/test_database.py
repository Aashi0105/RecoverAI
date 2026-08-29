"""
Unit & Integration Test Suite for RecoverAI Database Layer.
Uses an isolated in-memory SQLite database so tests NEVER depend on live PostgreSQL.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.database import Base
from database.models import FailedPayment, RecoveryAction, AuditLog
from database.repository import (
    save_recovery_audit,
    get_audit_by_transaction_id,
    list_recent_audits
)


@pytest.fixture
def db_session():
    """Provides isolated in-memory SQLite database session for testing."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


def test_database_table_creation(db_session):
    """Verify ORM models compile and database tables can be queried."""
    payments_count = db_session.query(FailedPayment).count()
    actions_count = db_session.query(RecoveryAction).count()
    audits_count = db_session.query(AuditLog).count()
    assert payments_count == 0
    assert actions_count == 0
    assert audits_count == 0


def test_save_recovery_audit(db_session):
    """Verify save_recovery_audit persists FailedPayment, RecoveryAction, and AuditLog records."""
    agent_result = {
        "transaction_id": "txn_db_test_001",
        "customer_id": "cust_db_001",
        "merchant_id": "merch_01",
        "amount": 2500.0,
        "currency": "INR",
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "recovery_probability": 0.85,
        "expected_recovery_value": 2125.0,
        "recommended_action": "retry",
        "policy_decision": "APPROVED",
        "policy_reason": "Approved by policy",
        "policy_violations": [],
        "selected_action": "retry",
        "action_status": "executed",
        "action_reference": "MOCK_RETRY_12345",
        "verification_status": "SUCCESS",
        "money_recovered": 2500.0,
        "agent_status": "COMPLETED",
        "audit_event": {
            "transaction_id": "txn_db_test_001",
            "policy_decision": "APPROVED",
            "selected_action": "retry"
        }
    }

    audit_entry = save_recovery_audit(db_session, agent_result)
    assert audit_entry is not None
    assert audit_entry.payment_id == "txn_db_test_001"

    # Verify FailedPayment row saved
    payment = db_session.query(FailedPayment).filter_by(id="txn_db_test_001").first()
    assert payment is not None
    assert payment.amount == 2500.0
    assert payment.recovery_probability == 0.85

    # Verify RecoveryAction row saved
    action = db_session.query(RecoveryAction).filter_by(payment_id="txn_db_test_001").first()
    assert action is not None
    assert action.action_type == "retry"
    assert action.result_status == "EXECUTED"

    # Verify AuditLog row saved
    audit_log = db_session.query(AuditLog).filter_by(payment_id="txn_db_test_001").first()
    assert audit_log is not None
    assert audit_log.details["policy_decision"] == "APPROVED"


def test_get_audit_by_transaction_id(db_session):
    """Verify get_audit_by_transaction_id retrieves persistent audit record."""
    txn_id = "txn_db_lookup_001"
    agent_result = {
        "transaction_id": txn_id,
        "amount": 1800.0,
        "recovery_probability": 0.75,
        "recommended_action": "retry",
        "policy_decision": "APPROVED",
        "selected_action": "retry",
        "agent_status": "COMPLETED",
        "audit_event": {
            "transaction_id": txn_id,
            "audit_key": "audit_value_123"
        }
    }

    save_recovery_audit(db_session, agent_result)
    fetched_audit = get_audit_by_transaction_id(db_session, txn_id)

    assert fetched_audit is not None
    assert fetched_audit["transaction_id"] == txn_id
    assert fetched_audit["audit_key"] == "audit_value_123"


def test_get_audit_missing_transaction_id(db_session):
    """Verify get_audit_by_transaction_id returns None for non-existent transaction ID."""
    result = get_audit_by_transaction_id(db_session, "non_existent_txn_999")
    assert result is None


def test_save_recovery_audit_idempotency(db_session):
    """Verify re-running save_recovery_audit for same transaction_id updates FailedPayment without primary key crash."""
    txn_id = "txn_db_idem_001"
    res1 = {
        "transaction_id": txn_id,
        "amount": 1000.0,
        "recovery_probability": 0.60,
        "agent_status": "PENDING",
        "audit_event": {"transaction_id": txn_id, "step": 1}
    }
    res2 = {
        "transaction_id": txn_id,
        "amount": 1000.0,
        "recovery_probability": 0.90,
        "agent_status": "COMPLETED",
        "audit_event": {"transaction_id": txn_id, "step": 2}
    }

    save_recovery_audit(db_session, res1)
    save_recovery_audit(db_session, res2)

    payment = db_session.query(FailedPayment).filter_by(id=txn_id).first()
    assert payment.recovery_probability == 0.90
    assert payment.status == "COMPLETED"

    audits = db_session.query(AuditLog).filter_by(payment_id=txn_id).all()
    assert len(audits) == 2


def test_list_recent_audits(db_session):
    """Verify list_recent_audits retrieves ordered recent audit records."""
    for i in range(3):
        agent_result = {
            "transaction_id": f"txn_recent_{i}",
            "amount": 1000.0 * (i + 1),
            "audit_event": {"transaction_id": f"txn_recent_{i}", "index": i}
        }
        save_recovery_audit(db_session, agent_result)

    recent = list_recent_audits(db_session, limit=2)
    assert len(recent) == 2
    assert "transaction_id" in recent[0]
