"""
Unit & Integration Test Suite for RecoverAI FastAPI REST API.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "recover-ai"


def test_trigger_recovery_valid_approved():
    """Verify POST /api/v1/recovery/trigger for high-confidence transient failure returns 200 OK APPROVED (ACT)."""
    payload = {
        "transaction_id": "txn_api_001",
        "customer_id": "cust_api_001",
        "merchant_id": "merch_01",
        "amount": 1500.0,
        "currency": "INR",
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "customer_historical_success_rate": 0.95,
        "previous_failures_24h": 0,
        "recovery_attempt_count": 0
    }
    response = client.post("/api/v1/recovery/trigger", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_api_001"
    assert data["policy_decision"] == "ACT"
    assert data["agent_status"] in ["APPROVED", "COMPLETED"]
    assert data["action_status"] == "executed"
    assert data["selected_action"] is not None
    assert "audit_event" in data


def test_trigger_recovery_blocked_policy():
    """Verify POST /api/v1/recovery/trigger for retry-limit-exceeded returns 200 OK REFUSE."""
    payload = {
        "transaction_id": "txn_api_002",
        "customer_id": "cust_api_002",
        "merchant_id": "merch_01",
        "amount": 2000.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "previous_failures_24h": 4,
        "consecutive_failure_streak": 4,
        "recovery_attempt_count": 4
    }
    response = client.post("/api/v1/recovery/trigger", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_api_002"
    assert data["policy_decision"] == "REFUSE"
    assert data["action_status"] == "not_executed"
    assert len(data["policy_violations"]) > 0


def test_trigger_recovery_human_approval():
    """Verify POST /api/v1/recovery/trigger for amount > ₹8,500 threshold returns 200 OK ESCALATE."""
    payload = {
        "transaction_id": "txn_api_003",
        "customer_id": "cust_api_003",
        "merchant_id": "merch_01",
        "amount": 10000.0,  # Exceeds ₹8,500 threshold
        "customer_average_transaction": 10000.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient",
        "customer_historical_success_rate": 0.95
    }
    response = client.post("/api/v1/recovery/trigger", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_api_003"
    assert data["policy_decision"] == "ESCALATE"
    assert data["agent_status"] == "AWAITING_APPROVAL"
    assert data["action_status"] == "not_executed"




def test_trigger_recovery_missing_required_fields():
    """Verify POST /api/v1/recovery/trigger with missing amount returns 422 Unprocessable Entity."""
    payload = {
        "transaction_id": "txn_api_004",
        "failure_reason": "network_timeout",
        "failure_category": "transient"
        # amount is missing!
    }
    response = client.post("/api/v1/recovery/trigger", json=payload)
    assert response.status_code == 422


def test_trigger_recovery_invalid_amount():
    """Verify POST /api/v1/recovery/trigger with invalid amount (<= 0) returns 422 Unprocessable Entity."""
    payload = {
        "transaction_id": "txn_api_005",
        "amount": -500.0,  # Invalid amount <= 0!
        "failure_reason": "network_timeout",
        "failure_category": "transient"
    }
    response = client.post("/api/v1/recovery/trigger", json=payload)
    assert response.status_code == 422


def test_audit_log_lookup_success():
    """Verify GET /api/v1/recovery/audit/{transaction_id} retrieves logged audit event."""
    txn_id = "txn_api_audit_001"
    payload = {
        "transaction_id": txn_id,
        "amount": 1200.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient"
    }
    # Trigger recovery to populate memory store
    trig_resp = client.post("/api/v1/recovery/trigger", json=payload)
    assert trig_resp.status_code == 200

    # Query audit lookup endpoint
    audit_resp = client.get(f"/api/v1/recovery/audit/{txn_id}")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["transaction_id"] == txn_id
    assert "policy_decision" in audit_data


def test_audit_log_lookup_not_found():
    """Verify GET /api/v1/recovery/audit/{transaction_id} returns 404 Not Found for unknown transaction."""
    response = client.get("/api/v1/recovery/audit/non_existent_txn_99999")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["error_code"] == "AUDIT_RECORD_NOT_FOUND"


def test_unexpected_agent_failure_handling():
    """Verify POST /api/v1/recovery/trigger handles internal errors with structured 500 without stack traces."""
    payload = {
        "transaction_id": "txn_api_fail",
        "amount": 1000.0,
        "failure_reason": "network_timeout",
        "failure_category": "transient"
    }
    with patch("backend.routes.recovery.run_agent", side_effect=RuntimeError("Simulated internal error")):
        response = client.post("/api/v1/recovery/trigger", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error_code"] == "AGENT_EXECUTION_ERROR"
        assert "Simulated internal error" not in str(data)  # Stack trace NOT leaked!
