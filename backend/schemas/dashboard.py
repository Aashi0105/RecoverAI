"""
Pydantic Schemas for RecoverAI Dashboard API Endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DashboardMetricsResponse(BaseModel):
    total_risk: float = Field(..., description="Total failed transaction volume at risk")
    formatted_total_risk: str = Field(..., description="INR formatted total risk")
    default_recovered: float = Field(..., description="Gross revenue recovered under industry default baseline (tau=0.50)")
    formatted_default_recovered: str = Field(..., description="INR formatted default recovered revenue")
    default_cost: float = Field(..., description="Action costs incurred under default baseline")
    default_net: float = Field(..., description="Realized net value under default baseline")
    opt_recovered: float = Field(..., description="Gross revenue recovered under RecoverAI EV-optimized policy (tau=0.35)")
    formatted_opt_recovered: str = Field(..., description="INR formatted EV recovered revenue")
    opt_cost: float = Field(..., description="Action costs incurred under RecoverAI EV policy")
    opt_net: float = Field(..., description="Realized net value under RecoverAI EV policy")
    gross_uplift_inr: float = Field(..., description="Gross revenue uplift (opt - default)")
    gross_uplift_pct: float = Field(..., description="Gross revenue uplift percentage of total risk")
    net_uplift_inr: float = Field(..., description="True net profit uplift (net_opt - net_default)")
    formatted_net_uplift_inr: str = Field(..., description="INR formatted true net profit uplift")
    net_uplift_pct: float = Field(..., description="True net profit uplift percentage of total risk")
    formatted_net_uplift_pct: str = Field(..., description="Formatted true net profit uplift percentage")
    inc_action_cost: float = Field(..., description="Incremental action cost invested")
    recovery_rate_pct: float = Field(..., description="Percentage of total risk captured")
    total_failed_count: int = Field(..., description="Total failed transactions in holdout")
    payments_selected: int = Field(..., description="Payments selected for recovery action under tau=0.35")
    pending_review_count: int = Field(..., description="Current count of pending human review escalations in database")


class FunnelStepResponse(BaseModel):
    step: int
    name: str
    count: int
    percentage: str
    badge: str
    description: str


class FunnelResponse(BaseModel):
    total_failed: int
    total_risk: float
    steps: List[FunnelStepResponse]
    summary: Dict[str, Any]


class DecisionItem(BaseModel):
    key: str
    name: str
    count: int
    percentage: float
    description: str
    color: str


class FailureCategoryItem(BaseModel):
    category: str
    share_pct: float
    recoverability: str


class DecisionDistributionResponse(BaseModel):
    holdout_size: int
    decisions: List[DecisionItem]
    failure_categories: List[FailureCategoryItem]


class RecentTransactionItem(BaseModel):
    id: str
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    failure_code: Optional[str] = None
    failure_reason: str
    status: str
    recovery_probability: Optional[float] = None
    created_at: Optional[str] = None


class RecentTransactionsResponse(BaseModel):
    total_records: int
    total_recovered: int
    total_blocked: int
    total_pending_approval: int
    transactions: List[RecentTransactionItem]


class DriftFeatureItem(BaseModel):
    feature: str
    type: str
    psi: float
    status: str
    recommendation: str


class DriftReportResponse(BaseModel):
    system_status: str
    significant_drift_count: int
    moderate_drift_count: int
    features: List[DriftFeatureItem]


class CalibrationResponse(BaseModel):
    threshold_comparison: List[Dict[str, Any]]
    default_threshold: float
    optimal_threshold: float
    development_grid_search: List[Dict[str, Any]]


class SensitivityResponse(BaseModel):
    source: str
    scenario_count: int
    scenarios: List[Dict[str, Any]]


class AuditLogResponse(BaseModel):
    source: str = Field(..., description="Description of the log file and population covered")
    total_records: int
    act_count: int
    escalate_count: int
    refuse_count: int
    decision_filter: Optional[str] = None
    filtered_count: int
    limit: int
    skip: int
    records: List[Dict[str, Any]]
