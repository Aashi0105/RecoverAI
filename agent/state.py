"""
Strongly Typed AgentState Definition for RecoverAI LangGraph Workflow.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Complete state dictionary passed through the LangGraph revenue recovery workflow.
    """
    # Identifiers
    transaction_id: str
    customer_id: str
    merchant_id: str

    # Payment Context
    amount: float
    currency: str
    payment_method: str
    payment_network: str
    payment_channel: str

    # Failure Details
    failure_reason: str
    failure_category: str

    # Structured Customer & Risk Context
    customer_context: Dict[str, Any]
    recovery_attempt_count: int
    customer_contacted_today: int
    previous_failures_24h: int
    previous_failures_7d: int
    consecutive_failure_streak: int
    customer_past_recovery_rate_pre_current: float
    ip_risk_score: float
    velocity_score: float


    # ML Output
    recovery_probability: float
    expected_recovery_value: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'

    # Diagnosis Output
    failure_diagnosis: Dict[str, Any]
    diagnosis_source: str  # 'llm', 'heuristic'

    # Recommendation Output
    recommended_action: str  # 'retry', 'payment_link', 'reminder', 'escalate', 'no_action'
    recommendation_reason: str
    recommendation_confidence: float
    recommendation_source: str  # 'llm', 'heuristic'
    recommendation_factors: List[str]
    recommendation_expected_benefit: Optional[str]

    # Deterministic Policy Guard Output
    policy_decision: str  # 'APPROVED', 'BLOCKED', 'HUMAN_APPROVAL'
    policy_reason: str
    policy_violations: List[str]
    decision_explanation: Optional[Dict[str, Any]]

    # Execution & Verification Output
    selected_action: Optional[str]
    action_parameters: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    money_recovered: float

    # Governance & Status
    audit_event: Optional[Dict[str, Any]]
    agent_status: str  # 'PENDING', 'ANALYZING', 'APPROVED', 'BLOCKED', 'AWAITING_APPROVAL', 'EXECUTED', 'VERIFIED', 'COMPLETED'
