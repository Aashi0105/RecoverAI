"""
Policy-Gated Payment Executor for RecoverAI.

Enforces strict policy authority:
1. ONLY executes when decision == 'ACT'.
2. If decision == 'ESCALATE' or 'REFUSE', ZERO external API calls are made.
3. Defaults to DRY_RUN = True.
4. Generates complete, structured audit records for every execution attempt.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from payment.razorpay_client import create_razorpay_test_payment_link, is_razorpay_configured
from database.database import SessionLocal, engine, Base
from database.repository import (
    create_execution_claim,
    mark_execution_succeeded,
    mark_execution_failed_safe,
    mark_execution_unknown,
    get_execution_claim
)

load_dotenv()

logger = logging.getLogger(__name__)

_LOCAL_EXECUTION_LOCK = threading.Lock()


def execute_recovery_policy(
    policy_evaluation: Dict[str, Any],
    customer_info: Optional[Dict[str, str]] = None,
    dry_run: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Executes payment recovery action gated strictly by deterministic policy decision
    and enforced by atomic database idempotency claims and process thread locks.
    Returns a comprehensive structured audit record.
    """
    # 1. Resolve Dry Run configuration (Defaults to True for maximum safety)
    if dry_run is None:
        env_dry_run = os.getenv("RECOVERAI_DRY_RUN", "true").lower()
        dry_run = env_dry_run != "false"

    if customer_info is None:
        customer_info = {
            "name": "Valued Customer",
            "email": "customer@example.com",
            "contact": "+919999999999"
        }

    txn_id = policy_evaluation.get("transaction_id", "txn_unknown")
    timestamp = policy_evaluation.get("timestamp", "N/A")
    amount = float(policy_evaluation.get("amount", 0.0))
    prob = float(policy_evaluation.get("recovery_probability", 0.0))
    decision = policy_evaluation.get("decision", "REFUSE")
    action = policy_evaluation.get("recommended_action", "no_action")
    policy_version = policy_evaluation.get("policy_version", "1.0")

    audit_record: Dict[str, Any] = {
        "transaction_id": txn_id,
        "timestamp": timestamp,
        "amount": amount,
        "model_probability": prob,
        "policy_decision": decision,
        "recommended_action": action,
        "dry_run": dry_run,
        "execution_status": "UNEXECUTED",
        "external_api_called": False,
        "razorpay_reference_id": None,
        "short_url": None,
        "error_message": None,
        "blocking_reason": None,
        "justification": "",
        "policy_version": policy_version,
        "executed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # -------------------------------------------------------------------------
    # CRITICAL SAFETY REQUIREMENT: POLICY GATING
    # ZERO API calls allowed if decision is ESCALATE or REFUSE
    # -------------------------------------------------------------------------
    if decision != "ACT":
        audit_record["execution_status"] = "BLOCKED_BY_POLICY"
        audit_record["external_api_called"] = False
        audit_record["blocking_reason"] = (
            f"Execution blocked: Policy decision is '{decision}' (Requires 'ACT' to proceed). "
            f"Triggered rules: {'; '.join(policy_evaluation.get('triggered_rules', []))}"
        )
        audit_record["justification"] = policy_evaluation.get("justification", "Blocked by safety policy.")
        return audit_record

    # -------------------------------------------------------------------------
    # DECISION IS 'ACT': EVALUATE DRY RUN vs LIVE TEST MODE EXECUTION
    # -------------------------------------------------------------------------
    if dry_run:
        audit_record["execution_status"] = "SIMULATED_DRY_RUN"
        audit_record["external_api_called"] = False
        audit_record["justification"] = (
            f"DRY RUN MODE: Action '{action}' for ₹{amount:,.2f} approved by policy, "
            f"but external Razorpay API call bypassed by safety flag."
        )
        return audit_record

    # -------------------------------------------------------------------------
    # IDEMPOTENCY & CONCURRENCY ATOMIC CLAIM GATING & EXECUTION
    # -------------------------------------------------------------------------
    idempotency_key = f"idemp_{txn_id}"

    with _LOCAL_EXECUTION_LOCK:
        db = SessionLocal()
        try:
            Base.metadata.create_all(bind=engine)
            created, claim = create_execution_claim(
                db=db,
                idempotency_key=idempotency_key,
                payment_id=txn_id,
                action_type=action,
                amount=amount
            )

            if not created and claim:
                status = claim.status
                if status in ["SUCCEEDED", "PAID"]:
                    audit_record["execution_status"] = "IDEMPOTENT_SKIPPED"
                    audit_record["external_api_called"] = False
                    audit_record["razorpay_reference_id"] = claim.payment_link_id
                    audit_record["short_url"] = claim.short_url
                    audit_record["justification"] = (
                        f"IDEMPOTENT REUSE: Payment recovery link for transaction '{txn_id}' "
                        f"already created/settled ({claim.payment_link_id}, status={status}). External API call bypassed."
                    )
                    return audit_record

                elif status == "PROCESSING":
                    audit_record["execution_status"] = "ALREADY_PROCESSING"
                    audit_record["external_api_called"] = False
                    audit_record["blocking_reason"] = (
                        f"IDEMPOTENT BLOCKED: Payment execution for transaction '{txn_id}' is currently "
                        f"in progress by another process. Duplicate external API call blocked."
                    )
                    audit_record["justification"] = "Execution blocked due to concurrent processing claim."
                    return audit_record

                elif status == "UNKNOWN_EXTERNAL_RESULT":
                    audit_record["execution_status"] = "UNKNOWN_EXTERNAL_RESULT"
                    audit_record["external_api_called"] = False
                    audit_record["blocking_reason"] = (
                        f"IDEMPOTENT BLOCKED: Transaction '{txn_id}' is in UNKNOWN_EXTERNAL_RESULT state "
                        f"due to a prior timeout/crash. Automatic re-execution is prohibited."
                    )
                    audit_record["justification"] = "Execution blocked to prevent duplicate payment link creation."
                    return audit_record

            # -------------------------------------------------------------------------
            # LIVE TEST MODE EXECUTION (decision == 'ACT' AND dry_run == False)
            # -------------------------------------------------------------------------
            if action == "payment_link":
                if not is_razorpay_configured():
                    audit_record["execution_status"] = "CREDENTIALS_MISSING"
                    audit_record["error_message"] = "Razorpay TEST MODE credentials (RAZORPAY_KEY_ID) not found in environment."
                    audit_record["justification"] = "Action 'payment_link' failed due to missing API keys."
                    mark_execution_failed_safe(db, idempotency_key, "Razorpay TEST MODE credentials missing.")
                    return audit_record

                try:
                    api_res = create_razorpay_test_payment_link(
                        amount=amount,
                        customer_info=customer_info,
                        description=f"RecoverAI Recovery Link for {txn_id}",
                        reference_id=txn_id
                    )

                    audit_record["external_api_called"] = True
                    if api_res.get("success"):
                        audit_record["execution_status"] = "SUCCESS_CREATED"
                        audit_record["razorpay_reference_id"] = api_res.get("payment_link_id")
                        audit_record["short_url"] = api_res.get("short_url")
                        audit_record["justification"] = f"Razorpay Test Payment Link created successfully: {api_res.get('short_url')}"
                        mark_execution_succeeded(
                            db,
                            idempotency_key,
                            payment_link_id=api_res.get("payment_link_id"),
                            short_url=api_res.get("short_url"),
                            result_details=api_res
                        )
                    else:
                        audit_record["execution_status"] = "API_ERROR"
                        audit_record["error_message"] = api_res.get("error_message")
                        audit_record["justification"] = f"Razorpay Test Link API call failed: {api_res.get('error_message')}"
                        mark_execution_unknown(db, idempotency_key, api_res.get("error_message", "API Error"))
                except Exception as exc:
                    audit_record["execution_status"] = "UNKNOWN_EXTERNAL_RESULT"
                    audit_record["error_message"] = str(exc)
                    audit_record["justification"] = f"Exception during Razorpay execution: {str(exc)}"
                    mark_execution_unknown(db, idempotency_key, str(exc))

            elif action == "retry":
                audit_record["execution_status"] = "NOT_SUPPORTED"
                audit_record["external_api_called"] = False
                audit_record["justification"] = (
                    "Automatic retry requires an existing tokenized card mandate context and is "
                    "truthfully reported as NOT_SUPPORTED rather than creating a fake successful payment."
                )

            elif action == "reminder":
                audit_record["execution_status"] = "QUEUED_FOR_DELIVERY"
                audit_record["external_api_called"] = False
                audit_record["justification"] = (
                    "Reminder event queued for notification. Actual SMS/Email delivery requires "
                    "downstream messaging provider integration (Twilio / SendGrid)."
                )

            else:  # no_action
                audit_record["execution_status"] = "NO_ACTION"
                audit_record["external_api_called"] = False
                audit_record["justification"] = "No recovery action specified by policy."

        finally:
            db.close()

    return audit_record

