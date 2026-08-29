"""
Database Repository Layer for RecoverAI.

Provides clean CRUD functions for persisting and querying failed payments,
recovery actions, and immutable audit logs in PostgreSQL / SQLite.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from database.models import FailedPayment, RecoveryAction, AuditLog, PaymentExecutionClaim


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
                executed_at=datetime.utcnow(),
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
            timestamp=datetime.utcnow()
        )
        db.add(audit_log)

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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
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
        claim.updated_at = datetime.utcnow()
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
        claim.updated_at = datetime.utcnow()
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
        claim.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(claim)
    return claim

