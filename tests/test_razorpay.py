"""
Unit Test Suite for Razorpay Test Mode Actions & Verification.
All Razorpay SDK calls are mocked to ensure tests NEVER make network requests or expose secrets.
"""

import pytest
from unittest.mock import MagicMock, patch
from agent.tools.razorpay_actions import (
    get_razorpay_client,
    generate_test_contact,
    razorpay_create_payment_link,
    razorpay_create_recovery_order,
    razorpay_fetch_payment_link_status,
    razorpay_fetch_order_status
)


def test_generate_test_contact_valid_non_repeating():
    """Verify generate_test_contact produces valid +91 non-repeating contact string."""
    contact1 = generate_test_contact("cust_123")
    contact2 = generate_test_contact("cust_987654321012")
    assert contact1.startswith("+91")
    assert len(contact1) == 13
    assert len(set(contact1[3:])) > 1  # Not all repeating digits!
    assert contact2 == "+91987654321012"[-13:] or contact2.startswith("+91")
from agent.tools.mock_actions import execute_mock_action
from agent.nodes.verification import verify_outcome
from backend.config import settings


def test_client_initialization_missing_credentials():
    """Verify get_razorpay_client raises ValueError when credentials are empty."""
    with patch.object(settings, "RAZORPAY_KEY_ID", ""), patch.object(settings, "RAZORPAY_KEY_SECRET", ""):
        with patch.dict("os.environ", {"RAZORPAY_KEY_ID": "", "RAZORPAY_KEY_SECRET": ""}):
            with pytest.raises(ValueError, match="missing or unconfigured"):
                get_razorpay_client()


def test_client_initialization_with_credentials():
    """Verify get_razorpay_client initializes razorpay.Client when credentials exist."""
    with patch.object(settings, "RAZORPAY_KEY_ID", "rzp_test_mockkey"), patch.object(settings, "RAZORPAY_KEY_SECRET", "mocksecret"):
        client = get_razorpay_client()
        assert client is not None


def test_create_payment_link_paise_conversion():
    """Verify razorpay_create_payment_link converts amount from INR to paise (1500.50 -> 150050)."""
    mock_client = MagicMock()
    mock_client.payment_link.create.return_value = {
        "id": "plink_test123",
        "short_url": "https://rzp.io/i/test123url",
        "status": "created"
    }

    with patch("agent.tools.razorpay_actions.get_razorpay_client", return_value=mock_client):
        res = razorpay_create_payment_link(
            transaction_id="txn_test_001",
            amount=1500.50,
            customer_id="cust_test_001"
        )

        assert res["status"] == "executed"
        assert res["action"] == "payment_link"
        assert res["provider"] == "razorpay"
        assert res["reference_id"] == "plink_test123"
        assert res["payment_link_url"] == "https://rzp.io/i/test123url"

        # Verify paise conversion
        mock_client.payment_link.create.assert_called_once()
        call_args = mock_client.payment_link.create.call_args[0][0]
        assert call_args["amount"] == 150050  # 1500.50 * 100
        assert call_args["currency"] == "INR"


def test_create_recovery_order_paise_conversion():
    """Verify razorpay_create_recovery_order converts amount from INR to paise (2500.00 -> 250000)."""
    mock_client = MagicMock()
    mock_client.order.create.return_value = {
        "id": "order_test999",
        "status": "created"
    }

    with patch("agent.tools.razorpay_actions.get_razorpay_client", return_value=mock_client):
        res = razorpay_create_recovery_order(
            transaction_id="txn_test_002",
            amount=2500.00
        )

        assert res["status"] == "executed"
        assert res["action"] == "retry"
        assert res["provider"] == "razorpay"
        assert res["reference_id"] == "order_test999"

        # Verify paise conversion
        mock_client.order.create.assert_called_once()
        call_args = mock_client.order.create.call_args[0][0]
        assert call_args["amount"] == 250000  # 2500.00 * 100
        assert call_args["currency"] == "INR"


def test_provider_failure_handling():
    """Verify razorpay_create_payment_link returns status 'failed' cleanly on API exception."""
    mock_client = MagicMock()
    mock_client.payment_link.create.side_effect = RuntimeError("Razorpay API Connection Error")

    with patch("agent.tools.razorpay_actions.get_razorpay_client", return_value=mock_client):
        res = razorpay_create_payment_link(
            transaction_id="txn_test_err",
            amount=1000.0,
            customer_id="cust_err"
        )

        assert res["status"] == "failed"
        assert res["provider"] == "razorpay"
        assert res["provider_status"] == "error"
        assert "Razorpay API Connection Error" in res["message"]


def test_mock_mode_fallback():
    """Verify execute_mock_action uses safe mock actions when RAZORPAY_ENABLED is False."""
    with patch.object(settings, "RAZORPAY_ENABLED", False):
        state = {
            "selected_action": "payment_link",
            "transaction_id": "txn_mock_001",
            "amount": 2000.0,
            "customer_id": "cust_mock_001"
        }
        res_state = execute_mock_action(state)
        action_res = res_state["action_result"]

        assert action_res["status"] == "executed"
        assert action_res["action"] == "payment_link"
        assert action_res["reference_id"].startswith("MOCK_PL_")
        assert "provider" not in action_res or action_res.get("provider") != "razorpay"


def test_real_mode_verification_returns_pending():
    """Verify real Razorpay order/link with status 'created' returns verification_status PENDING."""
    state = {
        "selected_action": "payment_link",
        "amount": 1500.0,
        "recovery_probability": 0.85,
        "failure_category": "transient",
        "action_result": {
            "status": "executed",
            "action": "payment_link",
            "reference_id": "plink_test999",
            "provider": "razorpay",
            "provider_status": "created"
        }
    }

    verified_state = verify_outcome(state)
    ver_res = verified_state["verification_result"]

    assert ver_res["verification_status"] == "PENDING"
    assert ver_res["payment_recovered"] is False
    assert ver_res["money_recovered"] == 0.0
    assert "Awaiting customer payment capture" in ver_res["result_reason"]
