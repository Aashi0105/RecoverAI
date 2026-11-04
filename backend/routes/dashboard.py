"""
Dashboard API Router for RecoverAI REST API.

Provides read-only operational and economic metrics to power modern React / external frontends:
1. GET /api/v1/dashboard/metrics — 4 core KPIs (Total Risk, Default Recovered, EV Recovered, True Net Uplift) + pending queue count.
2. GET /api/v1/dashboard/funnel — 5-step conversion funnel from failure ingestion to closed-loop settlement.
3. GET /api/v1/dashboard/decision-distribution — Policy breakdown (ACT, ESCALATE, REFUSE, NEGATIVE_EV) and failure categories.
4. GET /api/v1/dashboard/recent-transactions — Activity stream of real-time database transactions.
5. GET /api/v1/dashboard/drift — Population Stability Index (PSI) feature drift telemetry.
6. GET /api/v1/dashboard/calibration — Policy threshold comparison table and development grid search curve.
7. GET /api/v1/dashboard/sensitivity — 6-scenario financial stress testing grid from CSV.
8. GET /api/v1/dashboard/audit-log — Immutable decision audit trail with optional decision filtering.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from database.database import get_db
from database.models import ApprovalRequest
from backend.services.dashboard_data import (
    compute_live_top_metrics,
    get_recovery_funnel,
    get_decision_distribution,
    get_recent_transactions,
    get_drift_report,
    get_calibration_data,
    get_sensitivity_data,
    get_audit_log_data
)
from backend.schemas.dashboard import (
    DashboardMetricsResponse,
    FunnelResponse,
    DecisionDistributionResponse,
    RecentTransactionsResponse,
    DriftReportResponse,
    CalibrationResponse,
    SensitivityResponse,
    AuditLogResponse
)

router = APIRouter()


@router.get(
    "/metrics",
    response_model=DashboardMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Top Financial & Operational KPIs",
    description="Returns canonical top metrics comparing industry default baseline (tau=0.50) vs RecoverAI EV-optimal policy (tau=0.35) plus live pending review count."
)
def get_metrics(db: Session = Depends(get_db)):
    raw_metrics = compute_live_top_metrics()
    pending_count = db.query(ApprovalRequest).filter(
        ApprovalRequest.status.in_(["PENDING", "PENDING_APPROVAL"])
    ).count()

    return DashboardMetricsResponse(
        total_risk=raw_metrics["total_risk"],
        formatted_total_risk=f"₹{raw_metrics['total_risk']:,.2f}",
        default_recovered=raw_metrics["default_recovered"],
        formatted_default_recovered=f"₹{raw_metrics['default_recovered']:,.2f}",
        default_cost=raw_metrics["default_cost"],
        default_net=raw_metrics["default_net"],
        opt_recovered=raw_metrics["opt_recovered"],
        formatted_opt_recovered=f"₹{raw_metrics['opt_recovered']:,.2f}",
        opt_cost=raw_metrics["opt_cost"],
        opt_net=raw_metrics["opt_net"],
        gross_uplift_inr=raw_metrics["gross_uplift_inr"],
        gross_uplift_pct=raw_metrics["gross_uplift_pct"],
        net_uplift_inr=raw_metrics["net_uplift_inr"],
        formatted_net_uplift_inr=f"{'+' if raw_metrics['net_uplift_inr'] >= 0 else '-'}₹{abs(raw_metrics['net_uplift_inr']):,.2f}",
        net_uplift_pct=raw_metrics["net_uplift_pct"],
        formatted_net_uplift_pct=f"{'+' if raw_metrics['net_uplift_pct'] >= 0 else '-'}{abs(raw_metrics['net_uplift_pct']):.2f}%",
        inc_action_cost=raw_metrics["inc_action_cost"],
        recovery_rate_pct=raw_metrics["recovery_rate_pct"],
        total_failed_count=raw_metrics["total_failed_count"],
        payments_selected=raw_metrics["payments_selected"],
        pending_review_count=pending_count
    )


@router.get(
    "/funnel",
    response_model=FunnelResponse,
    status_code=status.HTTP_200_OK,
    summary="Get End-to-End Recovery Funnel",
    description="Returns the 5-stage conversion progression from ingestion through evaluation, policy action, and verified recovery."
)
def get_funnel():
    return get_recovery_funnel()


@router.get(
    "/decision-distribution",
    response_model=DecisionDistributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Policy Decision & Failure Category Distributions",
    description="Returns transaction outcome distribution (ACT, ESCALATE, REFUSE, NEGATIVE_EV) and categorical breakdown."
)
def get_decisions():
    return get_decision_distribution()


@router.get(
    "/recent-transactions",
    response_model=RecentTransactionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Recent Activity Feed",
    description="Returns recent failed payment records and processing statuses from the persistent relational database."
)
def get_activity_feed(
    limit: int = Query(10, ge=1, le=100, description="Max transaction records to return"),
    db: Session = Depends(get_db)
):
    return get_recent_transactions(db=db, limit=limit)


@router.get(
    "/drift",
    response_model=DriftReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Population Stability Index (PSI) Drift Report",
    description="Runs read-only PSI feature drift monitoring across categorical and numerical features without modifying disk audit logs."
)
def get_drift():
    return get_drift_report()


@router.get(
    "/calibration",
    response_model=CalibrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Threshold Calibration & Policy Comparison",
    description="Returns policy threshold comparisons and threshold search curve."
)
def get_calibration():
    return get_calibration_data()


@router.get(
    "/sensitivity",
    response_model=SensitivityResponse,
    status_code=status.HTTP_200_OK,
    summary="Get 6-Scenario Sensitivity Stress Test Results",
    description="Returns verified 6-scenario sensitivity analysis grid from CSV."
)
def get_sensitivity():
    return get_sensitivity_data()


@router.get(
    "/audit-log",
    response_model=AuditLogResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Immutable Decision Audit Trail",
    description="Returns chronological decision audit records with optional decision filtering (ACT, ESCALATE, REFUSE)."
)
def get_audit_trail(
    decision: Optional[str] = Query(None, description="Filter records by policy decision (ACT, ESCALATE, REFUSE)"),
    limit: int = Query(50, ge=1, le=200, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination")
):
    return get_audit_log_data(decision=decision, limit=limit, skip=skip)
