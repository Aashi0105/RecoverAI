"""
Pydantic Schemas for RecoverAI REST API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TransactionInputSchema(BaseModel):
    """Input payload representing a failed payment event."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: Optional[str] = Field("cust_001", description="Customer identifier")
    merchant_id: Optional[str] = Field("merch_001", description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    currency: Optional[str] = Field("INR", description="Currency code")
    payment_method: Optional[str] = Field("card", description="Payment method (card, upi, netbanking)")
    payment_network: Optional[str] = Field("visa", description="Payment network")
    payment_channel: Optional[str] = Field("web", description="Payment channel")
    failure_reason: str = Field(..., description="Raw payment failure reason code")
    failure_category: str = Field(..., description="Failure category (transient, customer_action_required, etc.)")

    # Optional Customer & Risk Features
    customer_age_days: Optional[int] = Field(180)
    customer_previous_transactions: Optional[int] = Field(10)
    customer_successful_transactions: Optional[int] = Field(8)
    customer_historical_success_rate: Optional[float] = Field(0.8)
    customer_lifetime_value: Optional[float] = Field(20000.0)
    customer_average_transaction: Optional[float] = Field(2000.0)
    customer_transaction_frequency_30d: Optional[int] = Field(3)
    amount_vs_customer_average: Optional[float] = Field(1.0)
    hour: Optional[int] = Field(14)
    day_of_week: Optional[int] = Field(2)
    is_weekend: Optional[int] = Field(0)
    is_subscription: Optional[int] = Field(0)
    subscription_age_days: Optional[int] = Field(0)
    is_first_transaction: Optional[int] = Field(0)
    previous_failures_24h: Optional[int] = Field(0)
    previous_failures_7d: Optional[int] = Field(0)
    previous_successes_30d: Optional[int] = Field(3)
    transactions_24h: Optional[int] = Field(1)
    transactions_7d: Optional[int] = Field(3)
    recovery_attempt_count: Optional[int] = Field(0)
    customer_contacted_today: Optional[int] = Field(0)
    ip_risk_score: Optional[float] = Field(0.08)
    velocity_score: Optional[float] = Field(0.12)
    device_changed: Optional[int] = Field(0)
    location_changed: Optional[int] = Field(0)


class RecoveryTriggerResponse(BaseModel):
    """Structured recovery execution response."""
    transaction_id: str
    amount: float
    currency: str = "INR"
    failure_reason: str
    failure_category: str
    recovery_probability: float
    expected_recovery_value: float
    diagnosis: Optional[str] = None
    recommended_action: str
    policy_decision: str
    policy_reason: str
    policy_violations: List[str] = []
    selected_action: Optional[str] = None
    action_status: str
    action_reference: Optional[str] = None
    verification_status: str
    money_recovered: float = 0.0
    agent_status: str
    audit_event: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized API Error Response."""
    status: str = "error"
    message: str
    error_code: str
