"""
Pydantic Request & Response Schemas for Human-in-the-Loop (HITL) Approvals API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ApprovalDecisionRequest(BaseModel):
    """Payload submitted by merchant when approving or rejecting a transaction."""
    notes: Optional[str] = Field(default=None, description="Optional merchant reasoning or operational notes")
    dry_run: bool = Field(default=False, description="If true, simulates execution without external API calls")


class ApprovalItemResponse(BaseModel):
    """Detailed representation of a single merchant approval record."""
    transaction_id: str
    merchant_id: str
    customer_id: str
    amount: float
    currency: str = "INR"
    failure_reason: Optional[str] = None
    failure_category: Optional[str] = None
    
    # ML & Strategy Recommendation
    recovery_probability: Optional[float] = None
    expected_recovery_value: Optional[float] = None
    recommended_action: str
    recommendation_reason: Optional[str] = None
    recommendation_confidence: Optional[float] = None
    recommendation_factors: Optional[List[str]] = None
    recommendation_expected_benefit: Optional[str] = None
    
    # Diagnosis & Policy Info
    diagnosis_summary: Optional[str] = None
    diagnosis_severity: Optional[str] = None
    policy_decision: str = "ESCALATE"
    policy_reason: Optional[str] = None
    
    # Approval Lifecycle & Resolution
    status: str
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    human_decision: Optional[str] = None
    human_notes: Optional[str] = None
    execution_details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ApprovalActionResponse(BaseModel):
    """Structured response returned after approving or rejecting an approval request."""
    message: str
    transaction_id: str
    status: str
    human_decision: Optional[str] = None
    human_notes: Optional[str] = None
    execution_details: Optional[Dict[str, Any]] = None


class PendingApprovalsResponse(BaseModel):
    """Summary of all pending approval requests awaiting merchant action."""
    total_pending: int
    approvals: List[ApprovalItemResponse]
