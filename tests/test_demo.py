"""
Test Suite for Phase 3C: Demo Simulator & Interactive Experience.

Verifies:
1. GET /api/v1/demo/scenarios lists all 4 predefined Buildathon scenarios.
2. Scenario A (Auto-Recovery): Executes end-to-end with Policy ACT and auto-action.
3. Scenario A Closed-Loop: Setting simulate_closed_loop=True simulates customer payment link settlement.
4. Scenario B (Human Approval): Policy ESCALATE triggers auto-queueing into ApprovalRequest.
5. Scenario C (Fraud Block): High-risk payment is blocked (Policy REFUSE), zero actions executed, and human approval blocked.
6. Scenario D (Low Probability): Negative EV payment is rationally refused (Policy REFUSE).
7. Invalid scenario name returns HTTP 400.
8. POST /api/v1/demo/simulate-webhook processes closed-loop settlement cleanly.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from database.database import Base, get_db
from database.models import FailedPayment, ApprovalRequest, RecoveryAction, AuditLog
from database.repository import get_approval_request

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session with StaticPool."""
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


def test_1_get_demo_scenarios(client):
    """Verifies GET /api/v1/demo/scenarios returns all 4 predefined demo scenarios."""
    response = client.get("/api/v1/demo/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    scenario_ids = [s["id"] for s in data]
    assert "auto_recovery" in scenario_ids
    assert "human_approval" in scenario_ids
    assert "fraud_block" in scenario_ids
    assert "low_probability" in scenario_ids


def test_2_simulate_auto_recovery_scenario(client, db_session):
    """Verifies Scenario A executes end-to-end with Policy ACT and auto-action."""
    payload = {"scenario": "auto_recovery", "simulate_closed_loop": False}
    response = client.post("/api/v1/demo/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["scenario"] == "auto_recovery"
    assert data["policy_decision"] == "ACT"
    assert data["recommended_action"] in ["retry", "payment_link"]
    assert data["action_status"] == "executed"
    assert len(data["timeline_steps"]) == 7

    # Should NOT be in approval queue
    approval_req = get_approval_request(db_session, data["transaction_id"])
    assert approval_req is None


def test_3_simulate_auto_recovery_with_closed_loop(client, db_session):
    """Verifies Scenario A with simulate_closed_loop=True confirms settlement to RECOVERED."""
    payload = {"scenario": "auto_recovery", "simulate_closed_loop": True}
    response = client.post("/api/v1/demo/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["policy_decision"] == "ACT"
    assert data["closed_loop_simulated"] is True
    assert data["verification_status"] == "SUCCESS"
    assert data["money_recovered"] == data["amount"]
    assert data["agent_status"] == "RECOVERED"


def test_4_simulate_human_approval_scenario(client, db_session):
    """Verifies Scenario B triggers Policy ESCALATE and enters Approval Queue."""
    payload = {"scenario": "human_approval"}
    response = client.post("/api/v1/demo/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["scenario"] == "human_approval"
    assert data["policy_decision"] == "ESCALATE"
    assert data["agent_status"] == "AWAITING_APPROVAL"
    assert data["action_status"] == "not_executed"

    # Must be persisted in ApprovalRequest queue!
    approval_req = get_approval_request(db_session, data["transaction_id"])
    assert approval_req is not None
    assert approval_req.status == "PENDING_APPROVAL"
    assert approval_req.amount == data["amount"]


def test_5_simulate_fraud_block_scenario(client, db_session):
    """Verifies Scenario C blocks high-risk payment with Policy REFUSE and zero actions."""
    payload = {"scenario": "fraud_block"}
    response = client.post("/api/v1/demo/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["scenario"] == "fraud_block"
    assert data["policy_decision"] == "REFUSE"
    assert data["action_status"] == "not_executed"
    assert data["money_recovered"] == 0.0

    # Human approval must be strictly prohibited (409 Conflict)
    approve_resp = client.post(f"/api/v1/approvals/{data['transaction_id']}/approve", json={})
    assert approve_resp.status_code == 409


def test_6_simulate_low_probability_scenario(client, db_session):
    """Verifies Scenario D rationally refuses low-probability recovery to protect merchant spend."""
    payload = {"scenario": "low_probability"}
    response = client.post("/api/v1/demo/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["scenario"] == "low_probability"
    assert data["policy_decision"] == "REFUSE"
    assert data["recovery_probability"] < 0.35
    assert data["expected_decision"] == "REFUSE"
    assert data["is_policy_matched"] is True
    assert data["action_status"] == "not_executed"
    assert data["money_recovered"] == 0.0
    assert "Negative EV" in data["business_title"]


def test_7_simulate_invalid_scenario_returns_400(client):
    """Verifies invalid scenario name returns HTTP 400 Bad Request."""
    response = client.post("/api/v1/demo/simulate", json={"scenario": "invalid_scenario_name"})
    assert response.status_code == 400


def test_8_simulate_webhook_closed_loop(client, db_session):
    """Verifies POST /api/v1/demo/simulate-webhook settles payment cleanly."""
    # First create a payment
    sim_resp = client.post("/api/v1/demo/simulate", json={"scenario": "auto_recovery"})
    txn_id = sim_resp.json()["transaction_id"]

    # Now simulate webhook
    webhook_resp = client.post(
        "/api/v1/demo/simulate-webhook",
        json={"transaction_id": txn_id, "event": "payment_link.paid"}
    )
    assert webhook_resp.status_code == 200
    data = webhook_resp.json()
    assert data["payment_status"] == "RECOVERED"


def test_9_scenario_replay_determinism_and_isolation(client, db_session):
    """
    Verifies that running scenarios A -> B -> C -> D -> A in sequence
    replays safely with zero state contamination and consistent policy verdicts.
    """
    sequence = [
        ("auto_recovery", "ACT"),
        ("human_approval", "ESCALATE"),
        ("fraud_block", "REFUSE"),
        ("low_probability", "REFUSE"),
        ("auto_recovery", "ACT")  # Replay A
    ]

    seen_txn_ids = set()
    for scenario_name, expected_policy in sequence:
        resp = client.post("/api/v1/demo/simulate", json={"scenario": scenario_name})
        assert resp.status_code == 200
        d = resp.json()
        assert d["policy_decision"] == expected_policy
        assert d["expected_decision"] == expected_policy
        assert d["is_policy_matched"] is True
        assert d["transaction_id"] not in seen_txn_ids, f"Duplicate txn_id {d['transaction_id']} on replay"
        seen_txn_ids.add(d["transaction_id"])

