from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, Integer, Boolean, Text, JSON
from database.database import Base


def get_utc_now():
    """Returns current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class FailedPayment(Base):
    """Stores incoming failed or at-risk payment events."""
    __tablename__ = "failed_payments"

    id = Column(String(50), primary_key=True, index=True)
    merchant_id = Column(String(50), index=True)
    customer_id = Column(String(50), index=True)
    customer_email = Column(String(100), nullable=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="INR")
    failure_code = Column(String(50), nullable=False)
    failure_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    
    # ML Score fields
    recovery_probability = Column(Float, nullable=True)
    status = Column(String(30), default="PENDING")  # PENDING, RECOVERING, RECOVERED, FAILED, ESCALATED


class RecoveryAction(Base):
    """Tracks recovery actions initiated by the LangGraph agent."""
    __tablename__ = "recovery_actions"

    id = Column(String(50), primary_key=True, index=True)
    payment_id = Column(String(50), nullable=False, index=True)
    action_type = Column(String(50), nullable=False)  # retry, payment_link, discount_nudge, human_escalation
    policy_checked = Column(Boolean, default=True)
    approved_by_human = Column(Boolean, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    result_status = Column(String(30), default="INITIATED")  # INITIATED, SUCCESS, FAILED
    result_details = Column(JSON, nullable=True)


class AuditLog(Base):
    """Immutable audit trail for complete agent execution transparent governance."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String(50), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)  # DIAGNOSIS, ML_SCORE, POLICY_CHECK, ACTION_TAKEN, VERIFICATION
    actor = Column(String(50), default="LANGGRAPH_AGENT")  # LANGGRAPH_AGENT, MERCHANT_HUMAN, SYSTEM
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=get_utc_now)


class PaymentExecutionClaim(Base):
    """
    Atomic execution claim table enforcing idempotency and state transitions.
    States: PROCESSING, SUCCEEDED, FAILED_SAFE, UNKNOWN_EXTERNAL_RESULT
    """
    __tablename__ = "payment_execution_claims"

    idempotency_key = Column(String(100), primary_key=True, index=True)
    payment_id = Column(String(50), nullable=False, index=True)
    status = Column(String(30), nullable=False, default="PROCESSING")  # PROCESSING, SUCCEEDED, FAILED_SAFE, UNKNOWN_EXTERNAL_RESULT
    action_type = Column(String(50), nullable=True)
    amount = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    # Execution result payload
    payment_link_id = Column(String(100), nullable=True)
    short_url = Column(String(255), nullable=True)
    result_details = Column(JSON, nullable=True)


class ApprovalRequest(Base):
    """
    Tracks transactions requiring Human-in-the-Loop (HITL) merchant approval.
    Lifecycle states:
    - PENDING_APPROVAL: Escalated by policy guard, waiting for merchant decision.
    - APPROVED_BY_HUMAN: Approved by merchant; transitions to EXECUTED upon completion.
    - REJECTED_BY_HUMAN: Rejected by merchant; zero payment actions permitted.
    - EXECUTED: Approved recovery action successfully executed.
    - EXECUTION_FAILED: Approved recovery action execution failed.
    """
    __tablename__ = "approval_requests"

    id = Column(String(50), primary_key=True, index=True)  # same as transaction_id
    transaction_id = Column(String(50), nullable=False, unique=True, index=True)
    merchant_id = Column(String(50), nullable=False, default="merch_001", index=True)
    customer_id = Column(String(50), nullable=False, default="cust_001", index=True)
    amount = Column(Float, nullable=False, default=0.0)
    currency = Column(String(10), default="INR")
    failure_reason = Column(String(255), nullable=True)
    failure_category = Column(String(50), nullable=True)
    
    # ML & Recommendation Metadata
    recovery_probability = Column(Float, nullable=True)
    expected_recovery_value = Column(Float, nullable=True)
    recommended_action = Column(String(50), nullable=False, default="payment_link")
    recommendation_reason = Column(Text, nullable=True)
    recommendation_confidence = Column(Float, nullable=True)
    recommendation_factors = Column(JSON, nullable=True)
    recommendation_expected_benefit = Column(Text, nullable=True)
    
    # Diagnosis & Policy Info
    diagnosis_summary = Column(Text, nullable=True)
    diagnosis_severity = Column(String(20), nullable=True)
    policy_decision = Column(String(30), nullable=False, default="ESCALATE")
    policy_reason = Column(Text, nullable=True)
    policy_violations = Column(JSON, nullable=True)
    
    # Approval Lifecycle & Resolution
    status = Column(String(30), nullable=False, default="PENDING_APPROVAL", index=True)
    created_at = Column(DateTime, default=get_utc_now)
    resolved_at = Column(DateTime, nullable=True)
    human_decision = Column(String(30), nullable=True)  # APPROVED, REJECTED
    human_notes = Column(Text, nullable=True)
    execution_details = Column(JSON, nullable=True)


