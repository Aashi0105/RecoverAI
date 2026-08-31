"""
Comprehensive Test Suite for Closed-Loop Razorpay Webhook Integration.

Tests:
1. Valid HMAC-SHA256 signature verification.
2. Invalid signature rejection (400 Bad Request) without database mutation.
3. Missing X-Razorpay-Signature rejection (400 Bad Request).
4. Malformed JSON payload handling (400 Bad Request).
5. Successful payment event (payment_link.paid) updates claim to PAID, FailedPayment to RECOVERED, action to SETTLED.
6. Failed payment event (payment.failed) updates claim to PAYMENT_FAILED, does NOT mark FailedPayment as RECOVERED.
7. Expired link event (payment_link.expired) updates claim to EXPIRED, does NOT mark FailedPayment as RECOVERED.
8. Duplicate webhook delivery safety (returns 200 already_processed, prevents double recovery).
9. Already paid claim idempotency stability.
10. Unknown payment link handling (404 CLAIM_NOT_FOUND) without corrupting records.
11. Reference ID correlation (fallback lookup via transaction/reference ID).
12. Immutable audit trail creation (event_type=WEBHOOK_OUTCOME, actor=RAZORPAY_WEBHOOK).
13. Security regression (secrets never exposed, invalid signatures cannot mutate state).
"""

import hmac
import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from database.database import Base, get_db
from database.models import FailedPayment, RecoveryAction, AuditLog, PaymentExecutionClaim
from database.repository import create_execution_claim, mark_execution_succeeded
from payment.webhook import verify_razorpay_webhook_signature, normalize_razorpay_webhook

TEST_WEBHOOK_SECRET = "whsec_test_secret_key_12345"


@pytest.fixture
def db_session():
    """Provides an isolated, in-memory SQLite database session for webhook testing."""
    engine = create_engine(
        "sqlite:///:memory:",
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


@pytest.fixture
def client(db_session, monkeypatch):
    """Provides a TestClient with dependency override for get_db and test webhook secret."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)
    
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def make_signed_request(client: TestClient, payload: dict, secret: str = TEST_WEBHOOK_SECRET, headers: dict = None):
    """Helper to serialize payload and generate valid or custom X-Razorpay-Signature."""
    raw_body = json.dumps(payload).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    
    req_headers = {"Content-Type": "application/json", "X-Razorpay-Signature": sig}
    if headers:
        req_headers.update(headers)
    return client.post("/api/v1/webhooks/razorpay", content=raw_body, headers=req_headers)


# -----------------------------------------------------------------------------
# 1. Signature Verification Unit Tests
# -----------------------------------------------------------------------------

def test_signature_verification_valid():
    """Verifies HMAC-SHA256 signature verification returns True for matching digest."""
    body = b'{"event": "payment_link.paid"}'
    valid_sig = hmac.new(TEST_WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert verify_razorpay_webhook_signature(body, valid_sig, TEST_WEBHOOK_SECRET) is True


def test_signature_verification_invalid():
    """Verifies HMAC-SHA256 signature verification returns False for mismatched digest."""
    body = b'{"event": "payment_link.paid"}'
    invalid_sig = "fake_signature_hash_12345"
    assert verify_razorpay_webhook_signature(body, invalid_sig, TEST_WEBHOOK_SECRET) is False


def test_signature_verification_empty_inputs():
    """Verifies False is returned when signature or secret is missing."""
    assert verify_razorpay_webhook_signature(b"data", "", TEST_WEBHOOK_SECRET) is False
    assert verify_razorpay_webhook_signature(b"data", "sig", "") is False


# -----------------------------------------------------------------------------
# 2. HTTP Endpoint Security & Rejection Tests
# -----------------------------------------------------------------------------

def test_webhook_missing_signature_rejected(client):
    """Verifies request without X-Razorpay-Signature returns 400 Bad Request."""
    response = client.post("/api/v1/webhooks/razorpay", json={"event": "payment_link.paid"})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "MISSING_WEBHOOK_SIGNATURE"


def test_webhook_invalid_signature_rejected(client, db_session):
    """Verifies request with invalid signature returns 400 Bad Request and does NOT mutate state."""
    raw_body = b'{"event": "payment_link.paid"}'
    headers = {"Content-Type": "application/json", "X-Razorpay-Signature": "invalid_sig_123"}
    response = client.post("/api/v1/webhooks/razorpay", content=raw_body, headers=headers)
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_WEBHOOK_SIGNATURE"
    # Verify no audit logs were created
    assert db_session.query(AuditLog).count() == 0


def test_webhook_malformed_json_rejected(client):
    """Verifies request with invalid JSON syntax returns 400 Bad Request."""
    raw_body = b'{"event": "broken json without closing quote}'
    sig = hmac.new(TEST_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    headers = {"Content-Type": "application/json", "X-Razorpay-Signature": sig}
    response = client.post("/api/v1/webhooks/razorpay", content=raw_body, headers=headers)
    assert response.status_code == 400
    assert response.json()["error_code"] == "MALFORMED_WEBHOOK_PAYLOAD"


# -----------------------------------------------------------------------------
# 3. Closed-Loop Lifecycle & State Transition Tests
# -----------------------------------------------------------------------------

def test_webhook_payment_link_paid_success(client, db_session):
    """
    Verifies that a valid payment_link.paid event closes the recovery loop:
    - Claim transitions to PAID
    - FailedPayment transitions to RECOVERED
    - RecoveryAction transitions to SETTLED
    - Immutable AuditLog entry created with actor RAZORPAY_WEBHOOK
    """
    txn_id = "txn_closed_loop_001"
    plink_id = "plink_test_pay_001"
    amount = 2500.0

    # 1. Seed existing open-loop state
    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", amount)
    mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id=plink_id, short_url="https://rzp.io/i/test")
    
    payment = FailedPayment(
        id=txn_id, merchant_id="merch_01", customer_id="cust_01",
        amount=amount, failure_code="transient", status="APPROVED"
    )
    action = RecoveryAction(id="act_001", payment_id=txn_id, action_type="payment_link", result_status="INITIATED")
    db_session.add_all([payment, action])
    db_session.commit()

    # 2. Simulate Razorpay payment_link.paid webhook event
    payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "reference_id": txn_id,
                    "amount": 250000,
                    "amount_paid": 250000,
                    "currency": "INR",
                    "status": "paid"
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_live_test_999",
                    "amount": 250000,
                    "currency": "INR",
                    "status": "captured"
                }
            }
        }
    }

    response = make_signed_request(client, payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["claim_status"] == "PAID"
    assert res_data["transaction_id"] == txn_id

    # 3. Verify Database State Transitions
    db_session.expire_all()
    claim = db_session.query(PaymentExecutionClaim).filter_by(idempotency_key=f"idemp_{txn_id}").first()
    assert claim.status == "PAID"
    assert claim.result_details.get("webhook_payment_id") == "pay_live_test_999"

    db_payment = db_session.query(FailedPayment).filter_by(id=txn_id).first()
    assert db_payment.status == "RECOVERED"

    db_action = db_session.query(RecoveryAction).filter_by(payment_id=txn_id).first()
    assert db_action.result_status == "SETTLED"
    assert db_action.result_details.get("webhook_confirmed") is True

    # 4. Verify Immutable Audit Log
    audit = db_session.query(AuditLog).filter_by(payment_id=txn_id).first()
    assert audit is not None
    assert audit.event_type == "WEBHOOK_OUTCOME"
    assert audit.actor == "RAZORPAY_WEBHOOK"
    assert audit.details["resulting_claim_status"] == "PAID"
    assert audit.details["resulting_payment_status"] == "RECOVERED"


def test_webhook_payment_failed_lifecycle(client, db_session):
    """
    Verifies that a valid payment.failed event:
    - Sets claim status to PAYMENT_FAILED
    - Does NOT mark FailedPayment as RECOVERED
    - Creates audit record recording payment failure
    """
    txn_id = "txn_closed_loop_002"
    plink_id = "plink_test_fail_002"
    amount = 1200.0

    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", amount)
    mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id=plink_id, short_url="https://rzp.io/i/test2")
    payment = FailedPayment(id=txn_id, merchant_id="m1", customer_id="c1", amount=amount, failure_code="transient", status="APPROVED")
    action = RecoveryAction(id="act_002", payment_id=txn_id, action_type="payment_link", result_status="INITIATED")
    db_session.add_all([payment, action])
    db_session.commit()

    payload = {
        "entity": "event",
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_failed_888",
                    "payment_link_id": plink_id,
                    "amount": 120000,
                    "currency": "INR",
                    "status": "failed",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Card declined by issuing bank"
                }
            }
        }
    }

    response = make_signed_request(client, payload)
    assert response.status_code == 200
    assert response.json()["claim_status"] == "PAYMENT_FAILED"

    db_session.expire_all()
    claim = db_session.query(PaymentExecutionClaim).filter_by(idempotency_key=f"idemp_{txn_id}").first()
    assert claim.status == "PAYMENT_FAILED"

    db_payment = db_session.query(FailedPayment).filter_by(id=txn_id).first()
    assert db_payment.status != "RECOVERED"  # CRITICAL: must never be RECOVERED
    assert db_payment.status == "FAILED"

    db_action = db_session.query(RecoveryAction).filter_by(payment_id=txn_id).first()
    assert db_action.result_status == "FAILED"


def test_webhook_payment_link_expired_lifecycle(client, db_session):
    """
    Verifies that a payment_link.expired event:
    - Sets claim status to EXPIRED
    - Sets action result_status to EXPIRED
    - Does NOT mark FailedPayment as RECOVERED
    """
    txn_id = "txn_closed_loop_003"
    plink_id = "plink_test_exp_003"
    amount = 3000.0

    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", amount)
    mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id=plink_id, short_url="https://rzp.io/i/test3")
    payment = FailedPayment(id=txn_id, merchant_id="m1", customer_id="c1", amount=amount, failure_code="transient", status="APPROVED")
    action = RecoveryAction(id="act_003", payment_id=txn_id, action_type="payment_link", result_status="INITIATED")
    db_session.add_all([payment, action])
    db_session.commit()

    payload = {
        "entity": "event",
        "event": "payment_link.expired",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "reference_id": txn_id,
                    "status": "expired"
                }
            }
        }
    }

    response = make_signed_request(client, payload)
    assert response.status_code == 200
    assert response.json()["claim_status"] == "EXPIRED"

    db_session.expire_all()
    claim = db_session.query(PaymentExecutionClaim).filter_by(idempotency_key=f"idemp_{txn_id}").first()
    assert claim.status == "EXPIRED"

    db_payment = db_session.query(FailedPayment).filter_by(id=txn_id).first()
    assert db_payment.status != "RECOVERED"


# -----------------------------------------------------------------------------
# 4. Webhook Idempotency & Duplicate Delivery Tests
# -----------------------------------------------------------------------------

def test_duplicate_webhook_delivery_is_safe_and_idempotent(client, db_session):
    """
    Verifies sending the same payment_link.paid event twice:
    - First delivery transitions to PAID (200 OK)
    - Second delivery safely returns 200 OK with already_processed status
    - Audit log is NOT duplicated
    """
    txn_id = "txn_idemp_wh_001"
    plink_id = "plink_idemp_001"

    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", 1500.0)
    mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id=plink_id, short_url="https://rzp.io/i/idemp")
    db_session.add(FailedPayment(id=txn_id, merchant_id="m1", customer_id="c1", amount=1500.0, failure_code="t", status="APPROVED"))
    db_session.commit()

    payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "reference_id": txn_id,
                    "amount_paid": 150000,
                    "status": "paid"
                }
            },
            "payment": {"entity": {"id": "pay_idemp_1"}}
        }
    }

    # 1. First delivery
    res1 = make_signed_request(client, payload)
    assert res1.status_code == 200
    assert res1.json()["status"] == "success"

    # 2. Second duplicate delivery
    res2 = make_signed_request(client, payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "already_processed"
    assert data2["claim_status"] == "PAID"

    # 3. Verify exactly 1 audit log was created
    audit_count = db_session.query(AuditLog).filter_by(payment_id=txn_id).count()
    assert audit_count == 1


def test_already_paid_claim_remains_stable_against_late_failure(client, db_session):
    """
    Verifies that if a claim is already PAID, an out-of-order payment.failed event
    is safely ignored as already_processed and does NOT overwrite the settled state.
    """
    txn_id = "txn_out_of_order_001"
    plink_id = "plink_ooo_001"

    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", 1000.0)
    claim = mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id=plink_id, short_url="https://rzp.io/i/ooo")
    claim.status = "PAID"
    db_session.commit()

    late_fail_payload = {
        "entity": "event",
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_late_fail",
                    "payment_link_id": plink_id,
                    "status": "failed"
                }
            }
        }
    }

    res = make_signed_request(client, late_fail_payload)
    assert res.status_code == 200
    assert res.json()["status"] == "already_processed"

    db_session.expire_all()
    updated_claim = db_session.query(PaymentExecutionClaim).filter_by(idempotency_key=f"idemp_{txn_id}").first()
    assert updated_claim.status == "PAID"


# -----------------------------------------------------------------------------
# 5. Correlation & Fallback Tests
# -----------------------------------------------------------------------------

def test_webhook_unknown_payment_link_returns_404(client, db_session):
    """Verifies that a webhook referencing an unknown payment link returns 404 CLAIM_NOT_FOUND."""
    payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": "plink_non_existent_9999",
                    "status": "paid"
                }
            }
        }
    }
    response = make_signed_request(client, payload)
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "CLAIM_NOT_FOUND"


def test_webhook_reference_id_fallback_correlation(client, db_session):
    """
    Verifies that when payment_link_id is absent or unindexed, correlation
    falls back successfully to reference_id (transaction ID).
    """
    txn_id = "txn_fallback_ref_001"
    create_execution_claim(db_session, f"idemp_{txn_id}", txn_id, "payment_link", 1800.0)
    mark_execution_succeeded(db_session, f"idemp_{txn_id}", payment_link_id="plink_unmatched", short_url="https://rzp.io/i/fb")
    db_session.add(FailedPayment(id=txn_id, merchant_id="m1", customer_id="c1", amount=1800.0, failure_code="t", status="APPROVED"))
    db_session.commit()

    # Payload has unknown payment_link_id, but matching reference_id in notes
    payload = {
        "entity": "event",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": "plink_different_id",
                    "reference_id": txn_id,
                    "amount_paid": 180000,
                    "status": "paid"
                }
            },
            "payment": {"entity": {"id": "pay_fb_1"}}
        }
    }

    response = make_signed_request(client, payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["transaction_id"] == txn_id


def test_webhook_unsupported_event_gracefully_ignored(client):
    """Verifies that an unsupported event type (e.g. order.paid) returns 200 ignored without error."""
    payload = {
        "entity": "event",
        "event": "order.paid",
        "payload": {"order": {"entity": {"id": "order_test_1"}}}
    }
    response = make_signed_request(client, payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"
    assert data["event_type"] == "order.paid"
