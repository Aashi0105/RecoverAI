"""
Pydantic Schemas for Demo Simulator API in RecoverAI.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DemoSimulationRequest(BaseModel):
    """Request payload for running a demo scenario simulation."""
    scenario: str = Field(
        default="auto_recovery",
        description="Demo scenario: 'auto_recovery', 'human_approval', 'fraud_block', 'low_probability'"
    )
    amount: Optional[float] = Field(default=None, description="Optional custom amount override")
    simulate_closed_loop: bool = Field(
        default=False,
        description="If true, simulates customer payment link completion via closed-loop webhook"
    )


class DemoWebhookSimulationRequest(BaseModel):
    """Request payload for manually simulating a closed-loop Razorpay webhook."""
    transaction_id: str = Field(..., description="Transaction ID to simulate webhook for")
    event: str = Field(default="payment_link.paid", description="Webhook event: payment_link.paid, payment.failed, payment_link.expired")


class DemoScenarioInfo(BaseModel):
    """Metadata describing a predefined demo scenario."""
    id: str
    name: str
    description: str
    expected_flow: str
    expected_decision: str
    expected_action: str
    amount: float
    failure_reason: str
    failure_category: str


class DemoSimulationResponse(BaseModel):
    """Full structured response returned by demo simulation."""
    scenario: str
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    currency: str
    failure_reason: str
    failure_category: str
    recovery_probability: float
    expected_recovery_value: float
    diagnosis: Optional[str] = None
    diagnosis_source: str = "heuristic"
    recommended_action: str
    recommendation_source: str = "heuristic"
    recommendation_confidence: Optional[float] = 0.8
    recommendation_factors: List[str] = Field(default_factory=list)
    recommendation_expected_benefit: Optional[str] = None
    policy_decision: str
    policy_reason: str
    policy_violations: List[str] = Field(default_factory=list)
    expected_decision: str = "ACT"
    is_policy_matched: bool = True
    business_title: str = ""
    business_impact: str = ""
    action_status: str
    selected_action: Optional[str] = None
    action_reference: Optional[str] = None
    verification_status: str
    money_recovered: float
    agent_status: str
    timeline_steps: List[Dict[str, Any]] = Field(default_factory=list)
    closed_loop_simulated: bool = False
