"""
Unit and Integration Tests for RecoverAI Dashboard API Endpoints (/api/v1/dashboard).
Fully isolated from physical development databases using in-memory SQLite fixtures.
"""

import os
import sys
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from database.database import Base, get_db
from database.models import FailedPayment, ApprovalRequest

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


def test_dashboard_metrics_canonical_values(client, db_session):
    """
    Verifies /api/v1/dashboard/metrics returns the canonical verified figures:
    Total Risk: ₹1,579,773.01
    Default Recovered: ₹889,907.57
    EV Recovered: ₹920,754.44
    True Net Uplift: +₹30,774.87 (+1.95%)
    Pending Reviews count (isolated from dev database)
    """
    # Seed deterministic pending approval record
    req = ApprovalRequest(
        id="appr_canonical_001",
        transaction_id="txn_canonical_001",
        merchant_id="merch_test_001",
        customer_id="cust_test_001",
        amount=14500.0,
        currency="INR",
        failure_reason="high_value",
        recommended_action="retry",
        recommendation_reason="High value order requiring confirmation",
        status="PENDING_APPROVAL"
    )
    db_session.add(req)
    db_session.commit()

    response = client.get("/api/v1/dashboard/metrics")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()

    # Exact canonical checks
    assert abs(data["total_risk"] - 1579773.01) < 0.01, f"Total risk mismatch: {data['total_risk']}"
    assert abs(data["default_recovered"] - 889907.57) < 0.01, f"Default recovered mismatch: {data['default_recovered']}"
    assert abs(data["opt_recovered"] - 920754.44) < 0.01, f"Opt recovered mismatch: {data['opt_recovered']}"
    assert abs(data["net_uplift_inr"] - 30774.87) < 0.01, f"Net uplift mismatch: {data['net_uplift_inr']}"
    assert round(data["net_uplift_pct"], 2) == 1.95, f"Net uplift pct mismatch: {data['net_uplift_pct']}"

    # Formatted string checks
    assert "1,579,773.01" in data["formatted_total_risk"]
    assert "889,907.57" in data["formatted_default_recovered"]
    assert "920,754.44" in data["formatted_opt_recovered"]
    assert "+₹30,774.87" == data["formatted_net_uplift_inr"] or "30,774.87" in data["formatted_net_uplift_inr"]
    assert "+1.95%" in data["formatted_net_uplift_pct"]

    # Operational metrics
    assert data["total_failed_count"] == 633
    assert data["payments_selected"] == 521
    assert data["pending_review_count"] == 1


def test_dashboard_funnel_endpoint(client):
    """Verifies 5-stage conversion funnel and summary aggregates."""
    response = client.get("/api/v1/dashboard/funnel")
    assert response.status_code == 200

    data = response.json()
    assert data["total_failed"] == 633
    assert abs(data["total_risk"] - 1579773.01) < 0.01

    steps = data["steps"]
    assert len(steps) == 5

    assert steps[0]["name"] == "Failed Ingest"
    assert steps[0]["count"] == 633
    assert steps[1]["name"] == "AI Analyzed"
    assert steps[1]["count"] == 633
    assert "Passes EV" in steps[2]["name"]
    assert steps[2]["count"] == 521
    assert steps[3]["name"] == "Executed"
    assert steps[3]["count"] == 503
    assert steps[4]["name"] == "Recovered"
    assert steps[4]["count"] == 362

    summary = data["summary"]
    assert summary["blocked_count"] == 112
    assert summary["escalated_count"] == 18
    assert summary["recovered_count"] == 362
    assert abs(summary["recovered_revenue"] - 920754.44) < 0.01


def test_dashboard_decision_distribution_endpoint(client):
    """Verifies single-pass decision distribution breakdown (ACT, ESCALATE, REFUSE, NEGATIVE_EV) and failure categories."""
    response = client.get("/api/v1/dashboard/decision-distribution")
    assert response.status_code == 200

    data = response.json()
    assert data["holdout_size"] == 633

    decisions = {d["key"]: d for d in data["decisions"]}
    assert set(decisions.keys()) == {"ACT", "ESCALATE", "REFUSE", "NEGATIVE_EV"}

    assert decisions["ACT"]["count"] == 503
    assert decisions["ESCALATE"]["count"] == 18
    assert decisions["REFUSE"]["count"] == 28
    assert decisions["NEGATIVE_EV"]["count"] == 84

    total_count = sum(d["count"] for d in data["decisions"])
    assert total_count == 633

    # Reconciles with the 521 payments passing EV threshold and 112 blocked/sub-threshold
    assert decisions["ACT"]["count"] + decisions["ESCALATE"]["count"] == 521
    assert decisions["REFUSE"]["count"] + decisions["NEGATIVE_EV"]["count"] == 112

    assert len(data["failure_categories"]) >= 5


def test_dashboard_reconciliation_invariants(client):
    """
    Asserts cross-endpoint consistency:
    1. /metrics payments_selected (521) == ACT (503) + ESCALATE (18)
    2. holdout_size == ACT (503) + ESCALATE (18) + REFUSE (28) + NEGATIVE_EV (84) == 633
    3. /funnel blocked_count (112) == REFUSE (28) + NEGATIVE_EV (84)
    4. /funnel recovered_count (362) recovered revenue == opt_recovered (₹920,754.44)
    """
    m_resp = client.get("/api/v1/dashboard/metrics")
    d_resp = client.get("/api/v1/dashboard/decision-distribution")
    f_resp = client.get("/api/v1/dashboard/funnel")

    assert m_resp.status_code == 200
    assert d_resp.status_code == 200
    assert f_resp.status_code == 200

    m_data = m_resp.json()
    d_data = d_resp.json()
    f_data = f_resp.json()

    d_map = {d["key"]: d["count"] for d in d_data["decisions"]}
    act = d_map["ACT"]
    esc = d_map["ESCALATE"]
    ref = d_map["REFUSE"]
    neg = d_map["NEGATIVE_EV"]

    # Invariant 1: Total test set size reconciliation
    assert act + esc + ref + neg == 633
    assert d_data["holdout_size"] == 633
    assert f_data["total_failed"] == 633
    assert m_data["total_failed_count"] == 633

    # Invariant 2: Candidate selection reconciliation (passing EV threshold tau=0.35)
    assert act + esc == 521
    assert m_data["payments_selected"] == 521
    assert f_data["steps"][2]["count"] == 521

    # Invariant 3: Blocked / refused reconciliation
    assert ref + neg == 112
    assert f_data["summary"]["blocked_count"] == 112

    # Invariant 4: Escalations reconciliation
    assert esc == 18
    assert f_data["summary"]["escalated_count"] == 18

    # Invariant 5: Recovered count and revenue consistency
    assert f_data["steps"][4]["count"] == 362
    assert f_data["summary"]["recovered_count"] == 362
    assert abs(f_data["summary"]["recovered_revenue"] - m_data["opt_recovered"]) < 0.01
    assert abs(f_data["summary"]["recovered_revenue"] - 920754.44) < 0.01


def test_dashboard_recent_transactions_endpoint(client, db_session):
    """Verifies activity feed retrieval from isolated test database."""
    # Seed deterministic synthetic payment records
    fp1 = FailedPayment(
        id="txn_test_001",
        merchant_id="merch_001",
        customer_id="cust_001",
        amount=2500.0,
        currency="INR",
        failure_code="ERR_TIMEOUT",
        failure_reason="network_timeout",
        status="RECOVERED",
        recovery_probability=0.88
    )
    fp2 = FailedPayment(
        id="txn_test_002",
        merchant_id="merch_001",
        customer_id="cust_002",
        amount=1500.0,
        currency="INR",
        failure_code="ERR_RISK",
        failure_reason="suspected_risk",
        status="BLOCKED",
        recovery_probability=0.12
    )
    db_session.add_all([fp1, fp2])
    db_session.commit()

    response = client.get("/api/v1/dashboard/recent-transactions?limit=5")
    assert response.status_code == 200

    data = response.json()
    assert data["total_records"] == 2
    assert data["total_recovered"] == 1
    assert data["total_blocked"] == 1
    assert data["total_pending_approval"] == 0
    assert isinstance(data["transactions"], list)
    assert len(data["transactions"]) == 2
    assert data["transactions"][0]["transaction_id"] in ["txn_test_001", "txn_test_002"]


def test_dashboard_drift_report_endpoint(client):
    """Verifies read-only PSI drift report endpoint."""
    response = client.get("/api/v1/dashboard/drift")
    assert response.status_code == 200

    data = response.json()
    assert "system_status" in data
    assert data["system_status"] == "HEALTHY"
    assert data["significant_drift_count"] == 0
    assert len(data["features"]) > 0

    feature_names = [f["feature"] for f in data["features"]]
    assert "amount" in feature_names
    assert "failure_reason" in feature_names


def test_dashboard_calibration_endpoint(client):
    """Verifies threshold calibration and grid search curve."""
    response = client.get("/api/v1/dashboard/calibration")
    assert response.status_code == 200

    data = response.json()
    assert len(data["threshold_comparison"]) == 3
    assert data["default_threshold"] == 0.50
    assert data["optimal_threshold"] == 0.35
    assert len(data["development_grid_search"]) > 0


def test_dashboard_sensitivity_endpoint(client):
    """Verifies 6-scenario sensitivity grid parsed from CSV."""
    response = client.get("/api/v1/dashboard/sensitivity")
    assert response.status_code == 200

    data = response.json()
    assert data["scenario_count"] == 6
    assert len(data["scenarios"]) == 6

    sc_names = [sc["Scenario"] for sc in data["scenarios"]]
    assert "1. BASE CASE" in sc_names
    assert "4. THRESHOLD +0.10 (tau=0.45)" in sc_names
    assert "6. PROBABILITY MISCALIBRATION (-0.10)" in sc_names

    # Check renamed non-causal column header exists
    assert "EV Realized Net Value (not causal)" in data["scenarios"][0]


def test_dashboard_audit_log_endpoint(client):
    """Verifies immutable audit trail querying, source metadata, and decision filtering."""
    # 1. Unfiltered
    response = client.get("/api/v1/dashboard/audit-log?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "decision_audit.jsonl" in data["source"]
    assert "total_records" in data
    assert "records" in data

    # 2. Filter by ACT
    resp_act = client.get("/api/v1/dashboard/audit-log?decision=ACT&limit=5")
    assert resp_act.status_code == 200
    act_data = resp_act.json()
    for rec in act_data["records"]:
        assert rec.get("safety", {}).get("decision") == "ACT"

    # 3. Filter by ESCALATE
    resp_esc = client.get("/api/v1/dashboard/audit-log?decision=ESCALATE&limit=5")
    assert resp_esc.status_code == 200
    esc_data = resp_esc.json()
    for rec in esc_data["records"]:
        assert rec.get("safety", {}).get("decision") == "ESCALATE"

    # 4. Filter by REFUSE
    resp_ref = client.get("/api/v1/dashboard/audit-log?decision=REFUSE&limit=5")
    assert resp_ref.status_code == 200
    ref_data = resp_ref.json()
    for rec in ref_data["records"]:
        assert rec.get("safety", {}).get("decision") == "REFUSE"


def test_cors_preflight_headers(client):
    """Verifies CORS headers for React dev server origins (5173 and 3000)."""
    # Test origin 5173
    response_5173 = client.options(
        "/api/v1/dashboard/metrics",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response_5173.status_code == 200
    assert response_5173.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # Test origin 3000
    response_3000 = client.options(
        "/api/v1/dashboard/metrics",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response_3000.status_code == 200
    assert response_3000.headers.get("access-control-allow-origin") == "http://localhost:3000"
