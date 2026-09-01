"""
Database Repository Layer for RecoverAI.

Provides clean CRUD functions for persisting and querying failed payments,
recovery actions, and immutable audit logs in PostgreSQL / SQLite.
"""

import uuid
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from database.models import FailedPayment, RecoveryAction, AuditLog, PaymentExecutionClaim, ApprovalRequest

_APPROVAL_LOCK = threading.Lock()


def save_recovery_audit(db: Session, agent_result: Dict[str, Any]) -> AuditLog:
    """
    Persists recovery workflow execution results atomically into database tables:
    - failed_payments
    - recovery_actions (if action selected)
    - audit_logs (immutable execution audit record)
    """
    txn_id = str(agent_result.get("transaction_id", f"txn_{uuid.uuid4().hex[:8]}"))
    audit_event = agent_result.get("audit_event") or agent_result

    try:
        # 1. Upsert FailedPayment record
        payment = db.query(FailedPayment).filter(FailedPayment.id == txn_id).first()
        if not payment:
            payment = FailedPayment(
                id=txn_id,
                merchant_id=str(agent_result.get("merchant_id", "merch_001")),
                customer_id=str(agent_result.get("customer_id", "cust_001")),
                amount=float(agent_result.get("amount", 0.0)),
                currency=str(agent_result.get("currency", "INR")),
                failure_code=str(agent_result.get("failure_category", "unknown")),
                failure_reason=str(agent_result.get("failure_reason", "unknown")),
                recovery_probability=float(agent_result.get("recovery_probability", 0.0)) if agent_result.get("recovery_probability") is not None else None,
                status=str(agent_result.get("agent_status", "COMPLETED"))
            )
            db.add(payment)
        else:
            payment.recovery_probability = float(agent_result.get("recovery_probability", 0.0)) if agent_result.get("recovery_probability") is not None else payment.recovery_probability
            payment.status = str(agent_result.get("agent_status", payment.status))

        # 2. Record RecoveryAction if selected
        selected_action = agent_result.get("selected_action")
        if selected_action:
            action_id = f"act_{uuid.uuid4().hex[:8]}"
            rec_action = RecoveryAction(
                id=action_id,
                payment_id=txn_id,
                action_type=selected_action,
                policy_checked=True,
                approved_by_human=(agent_result.get("policy_decision") == "HUMAN_APPROVAL"),
                executed_at=datetime.now(timezone.utc),
                result_status=str(agent_result.get("action_status", "INITIATED")).upper(),
                result_details={
                    "reference_id": agent_result.get("action_reference"),
                    "verification_status": agent_result.get("verification_status"),
                    "money_recovered": agent_result.get("money_recovered", 0.0)
                }
            )
            db.add(rec_action)

        # 3. Create AuditLog entry
        audit_log = AuditLog(
            payment_id=txn_id,
            event_type="RECOVERY_WORKFLOW",
            actor="LANGGRAPH_AGENT",
            details=audit_event,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(audit_log)

        # 4. If transaction requires human merchant review, queue into ApprovalRequest
        if agent_result.get("agent_status") == "AWAITING_APPROVAL" or agent_result.get("policy_decision") in ["ESCALATE", "HUMAN_APPROVAL"]:
            approval_req = db.query(ApprovalRequest).filter(ApprovalRequest.transaction_id == txn_id).first()
            diag = agent_result.get("failure_diagnosis") or {}
            diag_str = diag.get("diagnosis", str(diag)) if isinstance(diag, dict) else str(diag)
            diag_sev = diag.get("severity", "MEDIUM") if isinstance(diag, dict) else "MEDIUM"

            if not approval_req:
                approval_req = ApprovalRequest(
                    id=txn_id,
                    transaction_id=txn_id,
                    merchant_id=str(agent_result.get("merchant_id", "merch_001")),
                    customer_id=str(agent_result.get("customer_id", "cust_001")),
                    amount=float(agent_result.get("amount", 0.0)),
                    currency=str(agent_result.get("currency", "INR")),
                    failure_reason=str(agent_result.get("failure_reason", "unknown")),
                    failure_category=str(agent_result.get("failure_category", "unknown")),
                    recovery_probability=float(agent_result.get("recovery_probability", 0.0)) if agent_result.get("recovery_probability") is not None else None,
                    expected_recovery_value=float(agent_result.get("expected_recovery_value", 0.0)) if agent_result.get("expected_recovery_value") is not None else None,
                    recommended_action=str(agent_result.get("recommended_action", "payment_link")),
                    recommendation_reason=str(agent_result.get("recommendation_reason", "")),
                    recommendation_confidence=float(agent_result.get("recommendation_confidence", 0.8)) if agent_result.get("recommendation_confidence") is not None else None,
                    recommendation_factors=agent_result.get("recommendation_factors", []),
                    recommendation_expected_benefit=str(agent_result.get("recommendation_expected_benefit", "")),
                    diagnosis_summary=diag_str,
                    diagnosis_severity=diag_sev,
                    policy_decision=str(agent_result.get("policy_decision", "ESCALATE")),
                    policy_reason=str(agent_result.get("policy_reason", "")),
                    policy_violations=agent_result.get("policy_violations", []),
                    status="PENDING_APPROVAL",
                    created_at=datetime.now(timezone.utc)
                )
                db.add(approval_req)

        db.commit()
        db.refresh(audit_log)
        return audit_log

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Database error while saving recovery audit: {str(e)}") from e


def get_audit_by_transaction_id(db: Session, transaction_id: str) -> Optional[Dict[str, Any]]:
    """
    Queries persistent database for the latest audit log matching given transaction_id.
    Returns structured audit dictionary or None if not found.
    """
    log_entry = (
        db.query(AuditLog)
        .filter(AuditLog.payment_id == transaction_id)
        .order_by(AuditLog.timestamp.desc())
        .first()
    )

    if not log_entry:
        return None

    if isinstance(log_entry.details, dict):
        return log_entry.details
    elif isinstance(log_entry.details, str):
        try:
            return json.loads(log_entry.details)
        except Exception:
            return {"raw_details": log_entry.details}

    return None


def list_recent_audits(db: Session, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent audit records for dashboard analytics and audit review.
    """
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    results = []
    for log_entry in logs:
        if isinstance(log_entry.details, dict):
            results.append(log_entry.details)
        elif isinstance(log_entry.details, str):
            try:
                results.append(json.loads(log_entry.details))
            except Exception:
                results.append({"transaction_id": log_entry.payment_id, "raw_details": log_entry.details})
    return results


def get_execution_claim(db: Session, idempotency_key: str) -> Optional[PaymentExecutionClaim]:
    """Retrieves an execution claim by idempotency_key."""
    return db.query(PaymentExecutionClaim).filter(PaymentExecutionClaim.idempotency_key == idempotency_key).first()


def create_execution_claim(
    db: Session,
    idempotency_key: str,
    payment_id: str,
    action_type: str,
    amount: float
) -> Tuple[bool, Optional[PaymentExecutionClaim]]:
    """
    Attempts atomic INSERT of a PROCESSING claim.
    Returns (created_successfully: bool, claim_record: PaymentExecutionClaim).
    If IntegrityError occurs, rollbacks safely and returns (False, existing_claim).
    """
    from sqlalchemy.exc import IntegrityError
    claim = PaymentExecutionClaim(
        idempotency_key=idempotency_key,
        payment_id=payment_id,
        action_type=action_type,
        amount=amount,
        status="PROCESSING",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    try:
        db.add(claim)
        db.commit()
        db.refresh(claim)
        return True, claim
    except IntegrityError:
        db.rollback()
        existing = get_execution_claim(db, idempotency_key)
        return False, existing
    except Exception as e:
        db.rollback()
        existing = get_execution_claim(db, idempotency_key)
        if existing:
            return False, existing
        raise RuntimeError(f"Database error creating execution claim: {str(e)}") from e



def mark_execution_succeeded(
    db: Session,
    idempotency_key: str,
    payment_link_id: Optional[str] = None,
    short_url: Optional[str] = None,
    result_details: Optional[Dict[str, Any]] = None
) -> Optional[PaymentExecutionClaim]:
    claim = get_execution_claim(db, idempotency_key)
    if claim:
        claim.status = "SUCCEEDED"
        claim.payment_link_id = payment_link_id
        claim.short_url = short_url
        claim.result_details = result_details
        claim.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(claim)
    return claim


def mark_execution_failed_safe(
    db: Session,
    idempotency_key: str,
    reason: str
) -> Optional[PaymentExecutionClaim]:
    claim = get_execution_claim(db, idempotency_key)
    if claim:
        claim.status = "FAILED_SAFE"
        claim.result_details = {"blocking_reason": reason}
        claim.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(claim)
    return claim


def mark_execution_unknown(
    db: Session,
    idempotency_key: str,
    error_message: str
) -> Optional[PaymentExecutionClaim]:
    claim = get_execution_claim(db, idempotency_key)
    if claim:
        claim.status = "UNKNOWN_EXTERNAL_RESULT"
        claim.result_details = {"error_message": error_message}
        claim.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(claim)
    return claim


def get_execution_claim_by_link_id(
    db: Session,
    payment_link_id: str
) -> Optional[PaymentExecutionClaim]:
    """Retrieves an execution claim by Razorpay payment_link_id (Primary correlation)."""
    if not payment_link_id:
        return None
    return (
        db.query(PaymentExecutionClaim)
        .filter(PaymentExecutionClaim.payment_link_id == payment_link_id)
        .first()
    )


def get_execution_claim_by_payment_id(
    db: Session,
    payment_id: str
) -> Optional[PaymentExecutionClaim]:
    """Retrieves an execution claim by internal transaction/payment_id (Fallback correlation)."""
    if not payment_id:
        return None
    return (
        db.query(PaymentExecutionClaim)
        .filter(PaymentExecutionClaim.payment_id == payment_id)
        .first()
    )


def process_webhook_lifecycle_event(
    db: Session,
    event: Any
) -> Dict[str, Any]:
    """
    Correlates an incoming normalized Razorpay webhook event with internal records,
    checks idempotency, executes atomic state transitions across claims, payments,
    and actions, and creates an audit log entry.
    """
    event_type = getattr(event, "event_type", None)
    payment_link_id = getattr(event, "payment_link_id", None)
    reference_id = getattr(event, "reference_id", None)
    payment_id = getattr(event, "payment_id", None)
    amount = getattr(event, "amount", None)

    # 1. Check if event is supported by revenue recovery system
    if event_type not in ["payment_link.paid", "payment.failed", "payment_link.expired"]:
        return {
            "status": "ignored",
            "event_type": event_type,
            "message": f"Event '{event_type}' is unhandled by recovery webhook service and safely acknowledged."
        }

    # 2. Correlate with internal execution claim
    claim = None
    if payment_link_id:
        claim = get_execution_claim_by_link_id(db, payment_link_id)
    if not claim and reference_id:
        claim = get_execution_claim_by_payment_id(db, reference_id)

    if not claim:
        return {
            "status": "unmatched",
            "message": f"No claim found for payment_link_id='{payment_link_id}' or reference_id='{reference_id}'",
            "event_type": event_type
        }

    txn_id = claim.payment_id
    failed_payment = db.query(FailedPayment).filter(FailedPayment.id == txn_id).first()
    recovery_action = db.query(RecoveryAction).filter(RecoveryAction.payment_id == txn_id).first()
    now = datetime.now(timezone.utc)

    try:
        if event_type == "payment_link.paid":
            # Idempotency guard
            if claim.status == "PAID":
                return {
                    "status": "already_processed",
                    "transaction_id": txn_id,
                    "claim_status": claim.status,
                    "message": f"Payment execution claim for '{txn_id}' is already marked PAID."
                }

            claim.status = "PAID"
            claim.updated_at = now
            details = claim.result_details or {}
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {"raw": details}
            details["webhook_payment_id"] = payment_id
            details["amount_recovered"] = amount or claim.amount
            details["settled_at"] = now.isoformat()
            claim.result_details = details

            if failed_payment:
                failed_payment.status = "RECOVERED"

            if recovery_action:
                recovery_action.result_status = "SETTLED"
                act_details = recovery_action.result_details or {}
                if isinstance(act_details, str):
                    try:
                        act_details = json.loads(act_details)
                    except Exception:
                        act_details = {"raw": act_details}
                act_details["webhook_confirmed"] = True
                act_details["payment_id"] = payment_id
                act_details["amount_recovered"] = amount or claim.amount
                recovery_action.result_details = act_details

            audit_log = AuditLog(
                payment_id=txn_id,
                event_type="WEBHOOK_OUTCOME",
                actor="RAZORPAY_WEBHOOK",
                details={
                    "webhook_event": event_type,
                    "payment_link_id": payment_link_id,
                    "payment_id": payment_id,
                    "amount_recovered": amount or claim.amount,
                    "resulting_claim_status": "PAID",
                    "resulting_payment_status": "RECOVERED",
                    "resulting_action_status": "SETTLED"
                },
                timestamp=now
            )
            db.add(audit_log)
            db.commit()

            return {
                "status": "success",
                "action": "payment_confirmed",
                "transaction_id": txn_id,
                "claim_status": "PAID",
                "amount_recovered": amount or claim.amount
            }

        elif event_type == "payment.failed":
            if claim.status == "PAID":
                return {
                    "status": "already_processed",
                    "transaction_id": txn_id,
                    "claim_status": claim.status,
                    "message": f"Payment '{txn_id}' was already settled as PAID; ignoring late payment failure."
                }
            if claim.status == "PAYMENT_FAILED":
                return {
                    "status": "already_processed",
                    "transaction_id": txn_id,
                    "claim_status": claim.status,
                    "message": f"Payment execution claim for '{txn_id}' is already marked PAYMENT_FAILED."
                }

            claim.status = "PAYMENT_FAILED"
            claim.updated_at = now
            if failed_payment:
                failed_payment.status = "FAILED"
            if recovery_action:
                recovery_action.result_status = "FAILED"

            audit_log = AuditLog(
                payment_id=txn_id,
                event_type="WEBHOOK_OUTCOME",
                actor="RAZORPAY_WEBHOOK",
                details={
                    "webhook_event": event_type,
                    "payment_link_id": payment_link_id,
                    "payment_id": payment_id,
                    "error_code": getattr(event, "error_code", None),
                    "error_description": getattr(event, "error_description", None),
                    "resulting_claim_status": "PAYMENT_FAILED"
                },
                timestamp=now
            )
            db.add(audit_log)
            db.commit()

            return {
                "status": "success",
                "action": "payment_failed_recorded",
                "transaction_id": txn_id,
                "claim_status": "PAYMENT_FAILED"
            }

        elif event_type == "payment_link.expired":
            if claim.status == "PAID":
                return {
                    "status": "already_processed",
                    "transaction_id": txn_id,
                    "claim_status": claim.status,
                    "message": f"Payment '{txn_id}' was already settled as PAID; ignoring link expiration."
                }
            if claim.status == "EXPIRED":
                return {
                    "status": "already_processed",
                    "transaction_id": txn_id,
                    "claim_status": claim.status,
                    "message": f"Payment execution claim for '{txn_id}' is already marked EXPIRED."
                }

            claim.status = "EXPIRED"
            claim.updated_at = now
            if recovery_action:
                recovery_action.result_status = "EXPIRED"

            audit_log = AuditLog(
                payment_id=txn_id,
                event_type="WEBHOOK_OUTCOME",
                actor="RAZORPAY_WEBHOOK",
                details={
                    "webhook_event": event_type,
                    "payment_link_id": payment_link_id,
                    "resulting_claim_status": "EXPIRED"
                },
                timestamp=now
            )
            db.add(audit_log)
            db.commit()

            return {
                "status": "success",
                "action": "payment_link_expired",
                "transaction_id": txn_id,
                "claim_status": "EXPIRED"
            }

        else:
            return {
                "status": "ignored",
                "event_type": event_type,
                "message": f"Event '{event_type}' is unhandled and ignored."
            }

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Database error during webhook event processing: {str(e)}") from e


# =============================================================================
# HUMAN-IN-THE-LOOP (HITL) APPROVAL WORKFLOW
# =============================================================================

def list_pending_approvals(
    db: Session,
    merchant_id: Optional[str] = None,
    limit: int = 50
) -> List[ApprovalRequest]:
    """Retrieves pending human approval requests ordered by creation time descending."""
    query = db.query(ApprovalRequest).filter(ApprovalRequest.status == "PENDING_APPROVAL")
    if merchant_id:
        query = query.filter(ApprovalRequest.merchant_id == merchant_id)
    return query.order_by(ApprovalRequest.created_at.desc()).limit(limit).all()


def get_approval_request(db: Session, transaction_id: str) -> Optional[ApprovalRequest]:
    """Retrieves approval request by transaction_id."""
    return db.query(ApprovalRequest).filter(ApprovalRequest.transaction_id == transaction_id).first()


def approve_recovery_action(
    db: Session,
    transaction_id: str,
    human_notes: Optional[str] = None,
    dry_run: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Approves a transaction awaiting merchant review and triggers controlled execution.
    Returns (success: bool, status_code_label: str, result_payload: Dict[str, Any]).

    Safety Invariants:
    1. Rejects unknown transactions (NOT_FOUND).
    2. Rejects transactions blocked by policy (CANNOT_APPROVE_BLOCKED).
    3. Idempotent for already approved transactions (ALREADY_APPROVED).
    4. Rejects transactions already rejected by merchant (ALREADY_REJECTED).
    5. Atomic execution claim prevents double-execution under concurrency.
    """
    from payment.executor import execute_recovery_policy

    with _APPROVAL_LOCK:
        approval_req = get_approval_request(db, transaction_id)
        now = datetime.now(timezone.utc)

        # 1. Validation & Safety checks
        if not approval_req:
            # Check if transaction exists in FailedPayment or AuditLog as blocked
            payment = db.query(FailedPayment).filter(FailedPayment.id == transaction_id).first()
            if payment:
                audit = db.query(AuditLog).filter(AuditLog.payment_id == transaction_id).first()
                audit_details = audit.details if audit and isinstance(audit.details, dict) else {}
                policy_dec = audit_details.get("policy_decision")
                if payment.status in ["BLOCKED", "REFUSE"] or policy_dec in ["REFUSE", "BLOCKED"]:
                    return False, "CANNOT_APPROVE_BLOCKED", {
                        "error": "CANNOT_APPROVE_BLOCKED",
                        "message": "Transaction was blocked by deterministic fraud/risk policy and cannot be approved by human."
                    }
            return False, "NOT_FOUND", {
                "error": "NOT_FOUND",
                "message": f"Approval request for transaction '{transaction_id}' was not found."
            }

        # Already rejected check
        if approval_req.status == "REJECTED_BY_HUMAN":
            return False, "ALREADY_REJECTED", {
                "error": "ALREADY_REJECTED",
                "message": f"Transaction '{transaction_id}' has already been rejected by merchant. Cannot approve."
            }

        # Already approved / executed check (Idempotent reuse)
        if approval_req.status in ["APPROVED_BY_HUMAN", "EXECUTED"]:
            return True, "ALREADY_APPROVED", {
                "message": f"Transaction '{transaction_id}' has already been approved.",
                "transaction_id": transaction_id,
                "status": approval_req.status,
                "human_decision": approval_req.human_decision,
                "human_notes": approval_req.human_notes,
                "execution_details": approval_req.execution_details
            }

        if approval_req.status != "PENDING_APPROVAL":
            return False, "INVALID_STATE", {
                "error": "INVALID_STATE",
                "message": f"Transaction '{transaction_id}' is in status '{approval_req.status}' and cannot be approved."
            }

        # 2. Mark as APPROVED_BY_HUMAN
        approval_req.status = "APPROVED_BY_HUMAN"
        approval_req.human_decision = "APPROVED"
        approval_req.human_notes = human_notes
        approval_req.resolved_at = now
        db.commit()
        db.refresh(approval_req)

        # 3. Controlled Execution via Payment Executor
        policy_eval = {
            "transaction_id": transaction_id,
            "customer_id": approval_req.customer_id,
            "amount": approval_req.amount,
            "failure_reason": approval_req.failure_reason,
            "failure_category": approval_req.failure_category,
            "recovery_probability": approval_req.recovery_probability or 0.5,
            "decision": "ACT",  # Human authorized!
            "recommended_action": approval_req.recommended_action,
            "justification": f"Human merchant approval granted. Notes: {human_notes or 'None'}"
        }

        try:
            exec_res = execute_recovery_policy(policy_eval, dry_run=dry_run)
        except Exception as e:
            exec_res = {
                "execution_status": "FAILED",
                "error": str(e)
            }

        exec_status = exec_res.get("execution_status", "FAILED")
        if exec_status in ["SUCCEEDED", "SIMULATED_DRY_RUN", "IDEMPOTENT_SKIPPED"]:
            approval_req.status = "EXECUTED"
        else:
            approval_req.status = "EXECUTION_FAILED"

        approval_req.execution_details = exec_res

        # 4. Update FailedPayment
        payment = db.query(FailedPayment).filter(FailedPayment.id == transaction_id).first()
        if payment:
            payment.status = "APPROVED"

        # 5. Upsert RecoveryAction
        action_id = f"act_{uuid.uuid4().hex[:8]}"
        rec_action = RecoveryAction(
            id=action_id,
            payment_id=transaction_id,
            action_type=approval_req.recommended_action,
            policy_checked=True,
            approved_by_human=True,
            executed_at=now,
            result_status="SUCCESS" if approval_req.status == "EXECUTED" else "FAILED",
            result_details=exec_res
        )
        db.add(rec_action)

        # 6. Append Immutable AuditLog
        audit_log = AuditLog(
            payment_id=transaction_id,
            event_type="MERCHANT_APPROVAL",
            actor="MERCHANT_HUMAN",
            details={
                "transaction_id": transaction_id,
                "decision": "APPROVED",
                "notes": human_notes,
                "action_executed": approval_req.recommended_action,
                "execution_status": exec_status,
                "execution_details": exec_res,
                "timestamp": now.isoformat()
            },
            timestamp=now
        )
        db.add(audit_log)
        db.commit()
        db.refresh(approval_req)

        return True, "APPROVED", {
            "message": f"Transaction '{transaction_id}' successfully approved and executed.",
            "transaction_id": transaction_id,
            "status": approval_req.status,
            "human_decision": "APPROVED",
            "human_notes": human_notes,
            "execution_details": exec_res
        }


def reject_recovery_action(
    db: Session,
    transaction_id: str,
    human_notes: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Rejects a transaction awaiting merchant review.
    Ensures zero payment recovery actions are executed.
    """
    with _APPROVAL_LOCK:
        approval_req = get_approval_request(db, transaction_id)
        now = datetime.now(timezone.utc)

        if not approval_req:
            return False, "NOT_FOUND", {
                "error": "NOT_FOUND",
                "message": f"Approval request for transaction '{transaction_id}' was not found."
            }

        # Already approved check
        if approval_req.status in ["APPROVED_BY_HUMAN", "EXECUTED"]:
            return False, "ALREADY_APPROVED", {
                "error": "ALREADY_APPROVED",
                "message": f"Transaction '{transaction_id}' has already been approved and executed. Cannot reject."
            }

        # Idempotent rejection check
        if approval_req.status == "REJECTED_BY_HUMAN":
            return True, "ALREADY_REJECTED", {
                "message": f"Transaction '{transaction_id}' is already rejected.",
                "transaction_id": transaction_id,
                "status": "REJECTED_BY_HUMAN",
                "human_decision": "REJECTED",
                "human_notes": approval_req.human_notes
            }

        # Mark as REJECTED_BY_HUMAN
        approval_req.status = "REJECTED_BY_HUMAN"
        approval_req.human_decision = "REJECTED"
        approval_req.human_notes = human_notes
        approval_req.resolved_at = now

        # Update FailedPayment
        payment = db.query(FailedPayment).filter(FailedPayment.id == transaction_id).first()
        if payment:
            payment.status = "REJECTED"

        # Append Immutable AuditLog
        audit_log = AuditLog(
            payment_id=transaction_id,
            event_type="MERCHANT_REJECTION",
            actor="MERCHANT_HUMAN",
            details={
                "transaction_id": transaction_id,
                "decision": "REJECTED",
                "notes": human_notes,
                "action_executed": None,
                "timestamp": now.isoformat()
            },
            timestamp=now
        )
        db.add(audit_log)
        db.commit()
        db.refresh(approval_req)

        return True, "REJECTED", {
            "message": f"Transaction '{transaction_id}' rejected. Zero payment recovery actions executed.",
            "transaction_id": transaction_id,
            "status": "REJECTED_BY_HUMAN",
            "human_decision": "REJECTED",
            "human_notes": human_notes
        }



