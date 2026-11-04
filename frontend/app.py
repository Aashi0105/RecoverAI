"""
RecoverAI — Merchant-Facing Revenue Recovery Command Center & Buildathon Experience.

A modern, enterprise-grade AI operations dashboard for autonomous payment recovery,
Expected-Value policy optimization, deterministic safety guardrails, and closed-loop Razorpay settlement.

Integrates 5 Core Operational Workspaces:
- Tab 1: Recovery Command Center (Live KPIs, EV economics, recovery funnel, AI decision distribution, activity feed)
- Tab 2: Demo Simulator (1-click Buildathon scenarios A, B, C, D with instant injection & boundary testing)
- Tab 3: Live Decision Trace (7-stage visual pipeline with LLM/Heuristic badges & zero CoT leakage)
- Tab 4: Merchant Approval Queue (Human-in-the-loop governance for high-value orders & fraud safety invariants)
- Tab 5: Recovery Insights & Governance (EV threshold curves, PSI feature drift monitor, immutable audit logs)
"""

import os
import sys
import json
import time
import uuid
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FAIL-LOUD IMPORT VALIDATION
try:
    from sklearn.model_selection import train_test_split
    from ml.features import add_engineered_features, APPROVED_MODEL_FEATURES
    from evaluation.business_metrics import evaluate_threshold_grid, evaluate_zero_intervention_baseline
    from agent.graph import run_agent
    from agent.demo_data import build_test_transaction
    from agent.nodes.policy import load_frozen_policy, HIGH_VALUE_TRANSACTION_THRESHOLD
    from payment.razorpay_client import is_razorpay_configured
    from monitoring.drift_detection import run_drift_report
    from database.database import SessionLocal, engine, Base
    import database.models
    from database.models import FailedPayment, ApprovalRequest, AuditLog, PaymentExecutionClaim, RecoveryAction
    from database.repository import (
        list_pending_approvals,
        get_approval_request,
        approve_recovery_action,
        reject_recovery_action,
        save_recovery_audit,
        process_webhook_lifecycle_event,
        create_execution_claim,
        mark_execution_succeeded
    )
    from payment.webhook import normalize_razorpay_webhook
    from backend.routes.demo import PREDEFINED_SCENARIOS, construct_timeline
    from backend.services.dashboard_data import (
        load_and_split_dataset as _shared_load_and_split_dataset,
        compute_live_top_metrics as _shared_compute_live_top_metrics,
        get_recovery_funnel,
        get_decision_distribution
    )
except Exception as e:
    st.error(f"FAIL-LOUD INTEGRATION ERROR: Failed to import RecoverAI backend modules.\n\nError: {e}")
    st.stop()

# Ensure database tables exist
try:
    Base.metadata.create_all(bind=engine)
except Exception:
    pass

st.set_page_config(
    page_title="RecoverAI — Revenue Recovery Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# GLOBAL FINTECH DESIGN SYSTEM & CSS (RAZORPAY AESTHETIC)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Light Theme Palette */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #FFFFFF;
        color: #0F172A;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Clean Sidebar - Soft Grey Fintech Style */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] > div {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #475569;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0F172A !important;
    }

    /* Hero KPI Cards */
    .metric-card-hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 16px;
        text-align: left;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.15s ease;
    }
    .metric-card-hero:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    }
    .metric-card-blue {
        border-top: 3px solid #2563EB;
    }
    .metric-card-green {
        border-top: 3px solid #16A34A;
    }
    .metric-card-purple {
        border-top: 3px solid #7C3AED;
    }
    .metric-card-amber {
        border-top: 3px solid #F59E0B;
    }

    .metric-hero-label {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-hero-val {
        color: #0F172A;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 4px;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-hero-sub {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .metric-hero-delta-green {
        color: #15803D;
        background-color: #DCFCE7;
        border: 1px solid #86EFAC;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 6px;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }

    /* Funnel Step Card */
    .funnel-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 12px;
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .funnel-step-title {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .funnel-step-val {
        color: #0F172A;
        font-size: 1.30rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .funnel-step-pct {
        color: #2563EB;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Semantic Status Badges - Fintech Pill Style */
    .badge-act {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-recovered {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-approved {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-escalate {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-refuse {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-llm {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
    }
    .badge-heuristic {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
    }

    /* Trace & Pipeline Step Cards */
    .step-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .step-card-act {
        border-left-color: #16A34A;
    }
    .step-card-escalate {
        border-left-color: #F59E0B;
    }
    .step-card-refuse {
        border-left-color: #DC2626;
    }

    /* Feed / Activity Stream Cards */
    .activity-item {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    /* Scenario Selector Cards */
    .scenario-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 14px;
        text-align: left;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.15s ease;
    }
    .scenario-box:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
    }
    .scenario-box-active {
        border: 2px solid #2563EB !important;
        background: #EFF6FF !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08) !important;
    }
    .scenario-title {
        font-weight: 700;
        color: #0F172A;
        font-size: 0.95rem;
        letter-spacing: -0.01em;
    }
    .scenario-subtitle {
        font-weight: 600;
        color: #334155;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .scenario-caption {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Semantic Status Indicators */
    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-act {
        background-color: #16A34A;
    }
    .dot-recovered {
        background-color: #16A34A;
    }
    .dot-approved {
        background-color: #F59E0B;
    }
    .dot-escalate {
        background-color: #F59E0B;
    }
    .dot-refuse {
        background-color: #DC2626;
    }
    .dot-muted {
        background-color: #94A3B8;
    }

    /* Agent Decision Timeline Styles */
    .timeline-container {
        margin-top: 16px;
        margin-bottom: 20px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
    }
    .timeline-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: border-color 0.15s ease;
    }
    .timeline-card:hover {
        border-color: #CBD5E1;
    }
    .timeline-card-completed {
        border-left: 4px solid #16A34A;
    }
    .timeline-card-escalated {
        border-left: 4px solid #F59E0B;
    }
    .timeline-card-blocked {
        border-left: 4px solid #DC2626;
    }
    .timeline-pill {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .pill-completed {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .pill-escalated {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .pill-blocked {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .timeline-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #0F172A;
    }
    .timeline-stage-num {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .timeline-content {
        margin-top: 6px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.45;
    }
    .timeline-meta {
        margin-top: 8px;
        font-size: 0.78rem;
        color: #64748B;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
    }

    /* Primary & Secondary Buttons - Modern Fintech Look */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 7px 16px !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #2563EB !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }

    /* Tabs Component Styling - Clean Enterprise Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #64748B !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        padding: 10px 4px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: transparent !important;
    }

    /* Clean Code Spans */
    code {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
        font-size: 0.84em !important;
    }
</style>
""", unsafe_allow_html=True)

# Timestamps for Cache Invalidation
csv_file_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
csv_mtime = os.path.getmtime(csv_file_path) if os.path.exists(csv_file_path) else 0.0

model_artifact_file = os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib")
model_mtime = os.path.getmtime(model_artifact_file) if os.path.exists(model_artifact_file) else 0.0


@st.cache_data
def load_and_split_dataset(csv_mtime: float = 0.0):
    """Loads raw transactions dataset and splits into dev / test holdout sets via shared backend service."""
    return _shared_load_and_split_dataset(csv_mtime=csv_mtime)


@st.cache_data
def compute_live_top_metrics(model_mtime: float = 0.0, csv_mtime: float = 0.0):
    """
    Computes rigorous top-level business and economic KPI metrics comparing
    baseline industry threshold (0.50) vs RecoverAI EV-Optimized threshold (0.35)
    via shared backend service.
    """
    return _shared_compute_live_top_metrics(model_mtime=model_mtime, csv_mtime=csv_mtime)


def render_agent_decision_timeline(result: Dict[str, Any]) -> None:
    """
    Renders an interactive, visual Agent Decision Timeline illustrating
    the 7-stage lifecycle of an autonomous payment recovery decision.
    """
    if not result:
        return

    st.markdown("#### Decision Pipeline Timeline")
    st.caption("Visual end-to-end lifecycle trace of the recovery decision pipeline.")

    # 1. Dynamic Extraction from actual agent result
    txn_id = result.get("transaction_id", "Unknown")
    amount = result.get("amount", 0.0)
    currency = result.get("currency", "INR")
    failure_reason = result.get("failure_reason", "unknown")
    failure_category = result.get("failure_category", "unknown")

    prob = result.get("recovery_probability")
    prob_str = f"{prob * 100:.1f}%" if prob is not None else "Not available"
    ev = result.get("expected_recovery_value")
    ev_str = f"₹{ev:,.2f}" if ev is not None else "Not available"

    diagnosis = result.get("diagnosis") or "Failure pattern analyzed."
    diag_source = result.get("diagnosis_source", "heuristic")
    diag_badge = "LLM Reasoning" if diag_source == "llm" else "Heuristic Fallback"

    rec_action = (result.get("recommended_action") or "no_action").upper()
    rec_source = result.get("recommendation_source", "heuristic")
    rec_badge = "LLM Strategy" if rec_source == "llm" else "Heuristic Fallback"
    rec_factors = result.get("recommendation_factors", [])
    rec_factors_str = ", ".join(rec_factors) if rec_factors else "Not available"

    decision = result.get("policy_decision", "REFUSE")
    policy_reason = result.get("policy_reason", "Evaluated against recovery policy rules.")
    policy_violations = result.get("policy_violations", [])

    action_status = str(result.get("action_status", "not_executed")).upper()
    selected_action = (result.get("selected_action") or rec_action).upper()
    action_ref = result.get("action_reference") or "Not available"

    audit_event = result.get("audit_event") or {}
    audit_ts = audit_event.get("timestamp", "Recorded")
    agent_status = result.get("agent_status", "COMPLETED")
    customer_id = audit_event.get("customer_id") or "Not available"

    # 2. Status & content logic for stages 5 and 6
    if decision == "ACT":
        s5_class = "timeline-card-completed"
        s5_pill = '<span class="timeline-pill pill-completed">Approved (ACT)</span>'
        s6_class = "timeline-card-completed"
        s6_pill = f'<span class="timeline-pill pill-completed">Executed ({action_status})</span>'
        s6_content = f"Recovery action <b>{selected_action}</b> executed safely in Test Mode / Dry Run."
        s6_meta = f"<span>Action: <code>{selected_action}</code></span><span>Ref: <code>{action_ref}</code></span><span>Mode: <code>Dry Run</code></span>"
    elif decision == "ESCALATE":
        s5_class = "timeline-card-escalated"
        s5_pill = '<span class="timeline-pill pill-escalated">Escalated (ESCALATE)</span>'
        s6_class = "timeline-card-escalated"
        s6_pill = '<span class="timeline-pill pill-escalated">Pending Approval</span>'
        s6_content = "Automated execution safely halted. Transaction queued for Merchant Human-in-the-Loop review."
        s6_meta = "<span>Queue: <code>Merchant Approval Queue</code></span><span>Status: <code>PENDING_APPROVAL</code></span>"
    else:  # REFUSE
        s5_class = "timeline-card-blocked"
        s5_pill = '<span class="timeline-pill pill-blocked">Blocked (REFUSE)</span>'
        s6_class = "timeline-card-blocked"
        s6_pill = '<span class="timeline-pill pill-blocked">Action Refused</span>'
        s6_content = "Recovery action deliberately refused to protect merchant from chargebacks or spare action fee burn."
        s6_meta = "<span>Actions Executed: <code>0</code></span><span>Action Cost Avoided: <code>Spared</code></span>"

    # 3. Build structured explanation breakdown for Stage 5 if available
    explanation_html = ""
    exp = result.get("decision_explanation")
    if exp and isinstance(exp, dict):
        p_factor = exp.get("primary_factor", "UNKNOWN")
        checks = exp.get("policy_checks", {})
        reasons_list = exp.get("reasons", [])

        check_labels = {
            "fraud_risk": "Fraud Risk",
            "instrument_status": "Instrument Status",
            "failure_streak": "Failure Streak",
            "recovery_viability": "Recovery Viability (P ≥ τ)",
            "value_threshold": "Value Limit",
            "confidence_band": "Confidence Band",
            "velocity_risk": "Velocity Risk"
        }
        check_pills = []
        for k, v in checks.items():
            label = check_labels.get(k, k.replace("_", " ").title())
            if v == "PASSED":
                c_pill = f'<span class="timeline-pill pill-completed" style="margin-right:6px; margin-bottom:4px;">{label}: PASSED</span>'
            elif v == "FAILED":
                c_pill = f'<span class="timeline-pill pill-blocked" style="margin-right:6px; margin-bottom:4px;">{label}: FAILED</span>'
            elif v == "ESCALATED":
                c_pill = f'<span class="timeline-pill pill-escalated" style="margin-right:6px; margin-bottom:4px;">{label}: ESCALATED</span>'
            else:
                c_pill = f'<span class="timeline-pill" style="margin-right:6px; margin-bottom:4px; background:#F1F5F9; color:#475569; border:1px solid #CBD5E1;">{label}: {v}</span>'
            check_pills.append(c_pill)

        pills_str = "".join(check_pills)
        reasons_html = "".join([f"<li>{r}</li>" for r in reasons_list])
        is_open = "open" if decision != "ACT" else ""

        explanation_html = f"""
        <details style="margin-top:10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 12px;" {is_open}>
            <summary style="font-size:0.82rem; font-weight:700; color:#2563EB; cursor:pointer;">
                Structured Decision Explanation &amp; Policy Checks Breakdown
            </summary>
            <div style="margin-top:8px; font-size:0.82rem; color:#334155;">
                <div style="margin-bottom:6px;">
                    <span style="color:#64748B;">Primary Decision Factor:</span> <code>{p_factor}</code>
                </div>
                <div style="margin-bottom:8px;">
                    <span style="color:#64748B;">Policy Checks:</span><br/>
                    <div style="margin-top:4px; display:flex; flex-wrap:wrap;">{pills_str}</div>
                </div>
                <div>
                    <span style="color:#64748B;">Evaluated Reasons:</span>
                    <ul style="margin:4px 0 0 16px; padding:0; line-height:1.4;">
                        {reasons_html}
                    </ul>
                </div>
            </div>
        </details>
        """

    # 4. Assemble 7 stages
    stages = [
        {
            "num": 1,
            "title": "Context Loaded",
            "class": "timeline-card-completed",
            "pill": '<span class="timeline-pill pill-completed">Completed</span>',
            "content": f"Ingested payment failure telemetry for transaction <code>{txn_id}</code>.",
            "meta": f"<span>Customer: <code>{customer_id}</code></span><span>Amount: <b>₹{amount:,.2f} {currency}</b></span><span>Reason: <code>{failure_reason}</code></span>"
        },
        {
            "num": 2,
            "title": "ML Recovery Prediction",
            "class": "timeline-card-completed",
            "pill": '<span class="timeline-pill pill-completed">Completed</span>',
            "content": f"Calibrated recovery probability estimated at <b>{prob_str}</b> (Expected Value: <b>{ev_str}</b>).",
            "meta": "<span>Model: <code>EXP_0 Logistic Regression</code></span><span>Cutoff Threshold: <code>&tau; = 0.35</code></span>"
        },
        {
            "num": 3,
            "title": "Failure Diagnosis",
            "class": "timeline-card-completed",
            "pill": f'<span class="timeline-pill pill-completed">Completed</span> &nbsp;<span class="badge-llm">{diag_badge}</span>' if diag_source == "llm" else f'<span class="timeline-pill pill-completed">Completed</span> &nbsp;<span class="badge-heuristic">{diag_badge}</span>',
            "content": diagnosis,
            "meta": f"<span>Category: <code>{failure_category}</code></span><span>Diagnosis Source: <code>{diag_source.upper()}</code></span>"
        },
        {
            "num": 4,
            "title": "AI Strategy Recommendation",
            "class": "timeline-card-completed",
            "pill": f'<span class="timeline-pill pill-completed">Completed</span> &nbsp;<span class="badge-llm">{rec_badge}</span>' if rec_source == "llm" else f'<span class="timeline-pill pill-completed">Completed</span> &nbsp;<span class="badge-heuristic">{rec_badge}</span>',
            "content": f"Proposed recovery strategy: <b>{rec_action}</b>.",
            "meta": f"<span>Factors: <code>{rec_factors_str}</code></span><span>Strategy Source: <code>{rec_source.upper()}</code></span>"
        },
        {
            "num": 5,
            "title": "Deterministic Policy Guard",
            "class": s5_class,
            "pill": s5_pill,
            "content": policy_reason + explanation_html,
            "meta": f"<span>Policy Decision: <b>{decision}</b></span><span>Authority: <code>Deterministic Rules (YAML)</code></span>" + (f"<span>Violations: <code>{len(policy_violations)}</code></span>" if policy_violations else "")
        },
        {
            "num": 6,
            "title": "Controlled Execution",
            "class": s6_class,
            "pill": s6_pill,
            "content": s6_content,
            "meta": s6_meta
        },
        {
            "num": 7,
            "title": "Audit Logging & State Persistence",
            "class": "timeline-card-completed",
            "pill": '<span class="timeline-pill pill-completed">Completed</span>',
            "content": "Full decision chain atomically recorded in database (<code>AuditLog</code> table).",
            "meta": f"<span>Agent Status: <code>{agent_status}</code></span><span>Timestamp: <code>{audit_ts[:19] if len(audit_ts)>=19 else audit_ts}</code></span>"
        }
    ]

    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    for stage in stages:
        st.markdown(f"""
        <div class="timeline-card {stage['class']}">
            <div class="timeline-header">
                <div>
                    <span class="timeline-stage-num">Stage {stage['num']}</span>
                    <span class="timeline-title">{stage['title']}</span>
                </div>
                <div>{stage['pill']}</div>
            </div>
            <div class="timeline-content">{stage['content']}</div>
            <div class="timeline-meta">{stage['meta']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


test_df, idx_dev, idx_test = load_and_split_dataset(csv_mtime=csv_mtime)
metrics = compute_live_top_metrics(model_mtime=model_mtime, csv_mtime=csv_mtime)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 10px 0; border-bottom:1px solid #E2E8F0; margin-bottom:14px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:28px; height:28px; background:linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); border-radius:6px; display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-weight:800; font-size:0.95rem;">
                R
            </div>
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;">RecoverAI</div>
        </div>
        <div style="font-size:0.78rem; color:#64748B; margin-top:4px; font-weight:500;">Revenue Recovery &amp; FinTech Governance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Engine status detection
    from agent.services.llm_service import is_llm_available
    has_real_llm = is_llm_available()
    diag_status = "Live LLM" if has_real_llm else "Fallback Mode"
    
    razorpay_status = is_razorpay_configured()
    razorpay_label = "Test Mode" if razorpay_status else "Mock Sandbox"
    
    st.markdown(f"""
    <div style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
        System Status
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; font-size:0.80rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">Agent Pipeline</span>
            <span style="color:#15803D; font-weight:600; background:#DCFCE7; padding:1px 6px; border-radius:4px; border:1px solid #86EFAC;">Online</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">Diagnosis Engine</span>
            <span style="color:#0F172A; font-weight:600;">{diag_status}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">Policy Engine</span>
            <span style="color:#15803D; font-weight:600; background:#DCFCE7; padding:1px 6px; border-radius:4px; border:1px solid #86EFAC;">Active</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">Razorpay Integration</span>
            <span style="color:#0F172A; font-weight:600;">{razorpay_label}</span>
        </div>
    </div>
    <div style="border-top:1px solid #E2E8F0; margin:14px 0 10px 0;"></div>
    <div style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
        Parameters
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; font-size:0.80rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">EV Threshold (&tau;)</span>
            <span style="color:#0F172A; font-weight:600;">0.35</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#475569;">High-Value Escalation</span>
            <span style="color:#0F172A; font-weight:600;">₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}</span>
        </div>
    </div>
    <div style="border-top:1px solid #E2E8F0; margin:14px 0 12px 0;"></div>
    <div style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Safety Invariant
    </div>
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; font-size:0.78rem; color:#475569; line-height:1.45; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
        <div style="color:#0F172A; font-weight:600; margin-bottom:4px;">Governance Rules:</div>
        1. LLM Recommends.<br/>
        2. Deterministic Policy Decides.<br/>
        3. Human Approves Escalations.<br/>
        <span style="color:#DC2626; font-size:0.74rem; display:block; margin-top:6px; font-weight:600;">Fraud and risk blocks cannot be overridden.</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("RecoverAI v2.4-prod | Buildathon Submission Build")

# -----------------------------------------------------------------------------
# MAIN HEADER (RAZORPAY STYLE LOGO & TAGLINE)
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; margin-bottom:18px; border-bottom:1px solid #E2E8F0;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:38px; height:38px; background:linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); border-radius:8px; display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-weight:800; font-size:1.15rem; box-shadow:0 2px 4px rgba(37,99,235,0.25);">
            R
        </div>
        <div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.45rem; font-weight:800; color:#0F172A; letter-spacing:-0.03em;">RecoverAI</span>
                <span style="background:#EFF6FF; color:#2563EB; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:9999px; border:1px solid #BFDBFE;">Enterprise v2.4</span>
            </div>
            <div style="font-size:0.84rem; color:#64748B; margin-top:2px; font-weight:500;">
                Predicts recovery probability, applies deterministic policy rules, and confirms settlement through verified webhooks.
            </div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; background:#F8FAFC; border:1px solid #E2E8F0; padding:6px 14px; border-radius:9999px;">
        <span style="width:8px; height:8px; border-radius:50%; background:#16A34A; display:inline-block; box-shadow:0 0 0 3px rgba(22,163,74,0.15);"></span>
        <span style="font-size:0.75rem; font-weight:700; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em;">System Operational</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# WORKSPACE TABS
# -----------------------------------------------------------------------------
tab_cmd, tab_sim, tab_trace, tab_approvals, tab_insights = st.tabs([
    "Recovery Command Center",
    "Demo Simulator",
    "Live Decision Trace",
    "Merchant Approval Queue",
    "Recovery Insights & Governance"
])

# Initialize session state variables
if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None
if "latest_timeline" not in st.session_state:
    st.session_state["latest_timeline"] = None
if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "auto_recovery"


# =============================================================================
# TAB 1: RECOVERY COMMAND CENTER (CLEAN FINTECH DASHBOARD)
# =============================================================================
with tab_cmd:
    # Query database for live pending count and recent activity
    db = SessionLocal()
    try:
        pending_approvals = list_pending_approvals(db)
        pending_count = len(pending_approvals)
        recent_payments = db.query(FailedPayment).order_by(FailedPayment.created_at.desc()).limit(10).all()
        total_db_records = db.query(FailedPayment).count()
        total_recovered_db = db.query(FailedPayment).filter(FailedPayment.status == "RECOVERED").count()
        total_blocked_db = db.query(FailedPayment).filter(FailedPayment.status == "BLOCKED").count()
    except Exception:
        pending_count = 0
        recent_payments = []
        total_db_records = 0
        total_recovered_db = 0
        total_blocked_db = 0
    finally:
        db.close()

    # 1. TOP KPI HERO SECTION (Total Risk, Default Recovered, EV Recovered, True Net Uplift, Pending Reviews)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card-hero metric-card-blue">
            <div class="metric-hero-label">Total Risk</div>
            <div class="metric-hero-val">₹{metrics['total_risk']:,.2f}</div>
            <div class="metric-hero-sub">N = {metrics['total_failed_count']} Failed Txns</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card-hero">
            <div class="metric-hero-label">Default Recovered</div>
            <div class="metric-hero-val">₹{metrics['default_recovered']:,.2f}</div>
            <div class="metric-hero-sub">Standard Baseline (&tau; = 0.50)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card-hero metric-card-purple">
            <div class="metric-hero-label">EV Recovered</div>
            <div class="metric-hero-val" style="color:#7C3AED;">₹{metrics['opt_recovered']:,.2f}</div>
            <div class="metric-hero-sub">RecoverAI Policy (&tau; = 0.35)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card-hero metric-card-green">
            <div class="metric-hero-label">True Net Uplift</div>
            <div class="metric-hero-val" style="color:#16A34A;">+₹{metrics['net_uplift_inr']:,.2f}</div>
            <div class="metric-hero-delta-green">+{metrics['net_uplift_pct']:.2f}% Net Gain</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        badge_color = "#D97706" if pending_count > 0 else "#16A34A"
        badge_text = f"{pending_count} Awaiting" if pending_count > 0 else "Queue Clear"
        st.markdown(f"""
        <div class="metric-card-hero {'metric-card-amber' if pending_count > 0 else ''}">
            <div class="metric-hero-label">Pending Reviews</div>
            <div class="metric-hero-val" style="color:{badge_color};">{pending_count}</div>
            <div class="metric-hero-sub">{badge_text} in HITL Queue</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. RECOVERY PERFORMANCE STORY & FUNNEL (2 Columns)
    story_col1, story_col2 = st.columns([1.2, 1.8])

    with story_col1:
        st.subheader("Recovery Performance Story")
        st.markdown(
            "Comparing baseline industry heuristic (**&tau; = 0.50**) vs. "
            "**RecoverAI EV-Optimized Policy (&tau; = 0.35)** on holdout evaluation data:"
        )
        
        comp_card_html = f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.88rem;">
                <span style="color:#64748B;">Default Standard (0.50):</span>
                <span style="font-weight:600; color:#334155;">₹{metrics['default_recovered']:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.88rem;">
                <span style="color:#0F172A; font-weight:600;">RecoverAI Optimized (0.35):</span>
                <span style="font-weight:700; color:#2563EB;">₹{metrics['opt_recovered']:,.2f}</span>
            </div>
            <div style="border-top:1px solid #E2E8F0; padding-top:10px; display:flex; justify-content:space-between; font-size:0.90rem;">
                <span style="color:#15803D; font-weight:600;">Net Merchant Profit Uplift:</span>
                <span style="color:#15803D; font-weight:700;">+₹{metrics['net_uplift_inr']:,.2f} (+{metrics['net_uplift_pct']:.2f}%)</span>
            </div>
        </div>
        """
        st.markdown(comp_card_html, unsafe_allow_html=True)
        st.caption("*Expected Value Optimization mathematically proves that threshold 0.35 captures recoverable revenue without overspending on action costs.*")

    with story_col2:
        st.subheader("End-to-End Recovery Funnel")
        st.markdown("Transaction lifecycle progression from failure ingestion to verified settlement:")
        
        funnel_data = get_recovery_funnel(metrics)
        steps = funnel_data["steps"]
        summary = funnel_data["summary"]

        fn1, fn2, fn3, fn4, fn5 = st.columns(5)
        cols = [fn1, fn2, fn3, fn4, fn5]
        for col, s in zip(cols, steps):
            with col:
                card_style = ""
                title_style = ""
                val_style = ""
                pct_style = ""
                if s["step"] == 3:
                    card_style = ' style="border-color:#BFDBFE; background:#EFF6FF;"'
                    title_style = ' style="color:#2563EB;"'
                    val_style = ' style="color:#2563EB;"'
                elif s["step"] == 4:
                    val_style = ' style="color:#0F172A;"'
                    pct_style = ' style="color:#64748B;"'
                elif s["step"] == 5:
                    card_style = ' style="border-color:#86EFAC; background:#DCFCE7;"'
                    title_style = ' style="color:#15803D;"'
                    val_style = ' style="color:#15803D;"'
                    pct_style = ' style="color:#15803D;"'

                st.markdown(f"""
                <div class="funnel-card"{card_style}>
                    <div class="funnel-step-title"{title_style}>{s['step']}. {s['name']}</div>
                    <div class="funnel-step-val"{val_style}>{s['count']}</div>
                    <div class="funnel-step-pct"{pct_style}>{s['badge']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.80rem; color:#475569; padding:4px 6px; background:#F8FAFC; border-radius:6px; border:1px solid #E2E8F0;'>"
            f"<span><b style='color:#DC2626;'>{summary['blocked_count']} Blocked</b> (Fraud / Negative EV)</span>"
            f"<span><b style='color:#D97706;'>{summary['escalated_count']} Escalated</b> (High-Value HITL)</span>"
            f"<span><b style='color:#15803D;'>₹{summary['recovered_revenue']:,.0f} Recovered</b></span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. DECISIONS OVERVIEW & FAILURE CATEGORY INSIGHTS
    dec_col, cat_col = st.columns(2)
    
    with dec_col:
        st.subheader("Decision Distribution (Test Holdout N=633)")
        st.markdown("Policy Guard breakdown ensuring safe, cost-optimized recovery:")
        
        dec_data = get_decision_distribution()
        d_c1, d_c2 = st.columns(2)
        dec_map = {d["key"]: d for d in dec_data.get("decisions", [])}
        act_d = dec_map.get("ACT", {})
        esc_d = dec_map.get("ESCALATE", {})
        ref_d = dec_map.get("REFUSE", {})
        neg_d = dec_map.get("NEGATIVE_EV", {})

        def _fmt_stat(stat_dict):
            if not stat_dict or "count" not in stat_dict:
                return "—", ""
            cnt = stat_dict.get("count")
            pct = stat_dict.get("percentage")
            cnt_str = f"{cnt:,}" if cnt is not None else "—"
            pct_str = f"({pct:.1f}%)" if pct is not None else ""
            return cnt_str, pct_str

        act_cnt_str, act_pct_str = _fmt_stat(act_d)
        esc_cnt_str, esc_pct_str = _fmt_stat(esc_d)
        ref_cnt_str, ref_pct_str = _fmt_stat(ref_d)
        neg_cnt_str, neg_pct_str = _fmt_stat(neg_d)

        with d_c1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {act_d.get('color', '#16A34A')}; padding:12px 14px; border-radius:8px; margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="color:#15803D; font-weight:700; font-size:0.85rem; letter-spacing:0.02em;">{act_d.get('name', 'AUTO RECOVERY (ACT)').upper()}</span>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin:2px 0;">{act_cnt_str} <span style="font-size:0.85rem; color:#64748B; font-weight:600;">{act_pct_str}</span></div>
                <span style="color:#64748B; font-size:0.78rem;">{act_d.get('description', '')}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {esc_d.get('color', '#F59E0B')}; padding:12px 14px; border-radius:8px; margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="color:#B45309; font-weight:700; font-size:0.85rem; letter-spacing:0.02em;">{esc_d.get('name', 'HUMAN REVIEW (ESCALATE)').upper()}</span>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin:2px 0;">{esc_cnt_str} <span style="font-size:0.85rem; color:#64748B; font-weight:600;">{esc_pct_str}</span></div>
                <span style="color:#64748B; font-size:0.78rem;">{esc_d.get('description', '')}</span>
            </div>
            """, unsafe_allow_html=True)
        with d_c2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {ref_d.get('color', '#DC2626')}; padding:12px 14px; border-radius:8px; margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="color:#B91C1C; font-weight:700; font-size:0.85rem; letter-spacing:0.02em;">{ref_d.get('name', 'POLICY BLOCKED (REFUSE)').upper()}</span>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin:2px 0;">{ref_cnt_str} <span style="font-size:0.85rem; color:#64748B; font-weight:600;">{ref_pct_str}</span></div>
                <span style="color:#64748B; font-size:0.78rem;">{ref_d.get('description', '')}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {neg_d.get('color', '#94A3B8')}; padding:12px 14px; border-radius:8px; margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="color:#64748B; font-weight:700; font-size:0.85rem; letter-spacing:0.02em;">{neg_d.get('name', 'NO ACTION (NEGATIVE EV)').upper()}</span>
                <div style="font-size:1.35rem; font-weight:800; color:#0F172A; margin:2px 0;">{neg_cnt_str} <span style="font-size:0.85rem; color:#64748B; font-weight:600;">{neg_pct_str}</span></div>
                <span style="color:#64748B; font-size:0.78rem;">{neg_d.get('description', '')}</span>
            </div>
            """, unsafe_allow_html=True)

    with cat_col:
        st.subheader("Failure Category Distribution")
        st.markdown("Top drivers of payment failure across transaction history:")
        
        cat_rows = []
        for c in dec_data["failure_categories"]:
            cat_rows.append({
                "Category": c["category"],
                "Share": c["share_pct"],
                "Recoverability": c["recoverability"]
            })
        cat_data = pd.DataFrame(cat_rows)
        st.dataframe(cat_data, width='stretch', hide_index=True)
        st.caption("*Derived from historical failed payment distributions in `data/raw/transactions.csv`.*")

    st.markdown("---")

    # 4. ACTIVE RECOVERY ACTIVITY STREAM (PERSISTED DATABASE)
    st.subheader("Active Recovery Stream (Persisted Database Records)")
    
    if recent_payments:
        for p in recent_payments:
            badge_html = '<span class="badge-refuse">BLOCKED</span>'
            if p.status == "RECOVERED":
                badge_html = '<span class="badge-recovered">RECOVERED</span>'
            elif p.status == "AWAITING_APPROVAL":
                badge_html = '<span class="badge-escalate">AWAITING APPROVAL</span>'
            elif p.status == "COMPLETED":
                badge_html = '<span class="badge-act">COMPLETED</span>'
            elif p.status == "REJECTED_BY_HUMAN":
                badge_html = '<span class="badge-refuse">REJECTED BY MERCHANT</span>'

            ts_str = p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else "Just now"
            prob_str = f"{p.recovery_probability*100:.1f}%" if p.recovery_probability else "Evaluated"

            with st.container():
                st.markdown(f"""
                <div class="activity-item">
                    <div>
                        <div style="font-weight:700; font-size:0.95rem; color:#0F172A;">
                            Transaction <code>{p.id}</code> &nbsp;|&nbsp; <span style="color:#2563EB;">₹{p.amount:,.2f} {p.currency}</span>
                        </div>
                        <div style="color:#64748B; font-size:0.82rem; margin-top:4px;">
                            Customer: <code>{p.customer_id}</code> &nbsp;•&nbsp; Reason: <b>{p.failure_reason or 'N/A'}</b> &nbsp;•&nbsp; Timestamp: {ts_str}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        {badge_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"Technical Inspection ({p.id})", expanded=False):
                    st.json({
                        "transaction_id": p.id,
                        "merchant_id": p.merchant_id,
                        "customer_id": p.customer_id,
                        "amount": p.amount,
                        "currency": p.currency,
                        "failure_reason": p.failure_reason,
                        "failure_code": p.failure_code,
                        "status": p.status,
                        "recovery_probability": p.recovery_probability,
                        "created_at": ts_str
                    })
    else:
        st.info("No live transactions recorded yet in this session. Switch to the Demo Simulator tab to inject test failures and observe live recovery in action.")


# =============================================================================
# TAB 2: DEMO SIMULATOR
# =============================================================================
with tab_sim:
    st.subheader("Demo Simulator")
    st.markdown(
        "Select a curated industry scenario below to trigger a live payment failure and watch the "
        "recovery decision pipeline execute in real time."
    )

    # 4 Scenario Selection Cards (Cards A-D)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        is_sel_a = (st.session_state["selected_scenario"] == "auto_recovery")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_a else ''}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="status-dot dot-act"></span>
                    <span class="scenario-title">Scenario A</span>
                </div>
                {'<span style="background:#2563EB; color:#FFFFFF; font-size:0.68rem; font-weight:700; padding:1px 6px; border-radius:4px;">ACTIVE</span>' if is_sel_a else ''}
            </div>
            <div class="scenario-subtitle">Smart Auto-Recovery</div>
            <div class="scenario-caption">Transient timeout &rarr; High P &rarr; Auto-executes &amp; settles</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario A", key="btn_sc_a", width='stretch'):
            st.session_state["selected_scenario"] = "auto_recovery"
            st.rerun()

    with sc2:
        is_sel_b = (st.session_state["selected_scenario"] == "human_approval")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_b else ''}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="status-dot dot-escalate"></span>
                    <span class="scenario-title">Scenario B</span>
                </div>
                {'<span style="background:#2563EB; color:#FFFFFF; font-size:0.68rem; font-weight:700; padding:1px 6px; border-radius:4px;">ACTIVE</span>' if is_sel_b else ''}
            </div>
            <div class="scenario-subtitle">High-Value HITL Review</div>
            <div class="scenario-caption">₹14,500 &gt; limit &rarr; Escalates to Merchant Queue</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario B", key="btn_sc_b", width='stretch'):
            st.session_state["selected_scenario"] = "human_approval"
            st.rerun()

    with sc3:
        is_sel_c = (st.session_state["selected_scenario"] == "fraud_block")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_c else ''}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="status-dot dot-refuse"></span>
                    <span class="scenario-title">Scenario C</span>
                </div>
                {'<span style="background:#2563EB; color:#FFFFFF; font-size:0.68rem; font-weight:700; padding:1px 6px; border-radius:4px;">ACTIVE</span>' if is_sel_c else ''}
            </div>
            <div class="scenario-subtitle">Fraud / Risk Block</div>
            <div class="scenario-caption">IP risk 0.92 &rarr; Policy blocks &rarr; Zero execution</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario C", key="btn_sc_c", width='stretch'):
            st.session_state["selected_scenario"] = "fraud_block"
            st.rerun()

    with sc4:
        is_sel_d = (st.session_state["selected_scenario"] == "low_probability")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_d else ''}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="status-dot dot-muted"></span>
                    <span class="scenario-title">Scenario D</span>
                </div>
                {'<span style="background:#2563EB; color:#FFFFFF; font-size:0.68rem; font-weight:700; padding:1px 6px; border-radius:4px;">ACTIVE</span>' if is_sel_d else ''}
            </div>
            <div class="scenario-subtitle">Negative EV Refusal</div>
            <div class="scenario-caption">P &lt; 0.35 &rarr; Negative EV &rarr; Spares action fee</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario D", key="btn_sc_d", width='stretch'):
            st.session_state["selected_scenario"] = "low_probability"
            st.rerun()

    current_scen_key = st.session_state["selected_scenario"]
    scen_meta = PREDEFINED_SCENARIOS[current_scen_key]

    st.markdown("---")
    
    # Active Scenario Card Details
    exp_decision = scen_meta.get("expected_decision", "ACT")
    if exp_decision == "ACT":
        exp_badge_color = "#15803D"
        exp_badge_bg = "#DCFCE7"
        exp_badge_border = "#86EFAC"
    elif exp_decision == "ESCALATE":
        exp_badge_color = "#B45309"
        exp_badge_bg = "#FEF3C7"
        exp_badge_border = "#FDE68A"
    else:
        exp_badge_color = "#B91C1C"
        exp_badge_bg = "#FEE2E2"
        exp_badge_border = "#FCA5A5"
    
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <span style="font-size:1.10rem; font-weight:700; color:#0F172A;">
                {scen_meta['name']}
            </span>
            <span style="color:#475569; font-size:0.85rem;">
                Target Policy: <span style="color:{exp_badge_color}; background:{exp_badge_bg}; font-weight:700; padding:3px 10px; border:1px solid {exp_badge_border}; border-radius:6px;">{exp_decision}</span> &nbsp;|&nbsp; Action: <code>{scen_meta['expected_action'].upper()}</code>
            </span>
        </div>
        <div style="color:#334155; font-size:0.90rem; margin-top:8px; line-height:1.45;">
            {scen_meta['description']}
        </div>
        <div style="margin-top:12px; padding:10px 14px; background:#F8FAFC; border-radius:6px; border:1px solid #E2E8F0; font-size:0.82rem; color:#475569; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
            <span><b>Reason:</b> <code>{scen_meta['failure_reason']}</code></span>
            <span><b>Category:</b> <code>{scen_meta['failure_category']}</code></span>
            <span><b>Context:</b> {scen_meta.get('business_title', 'Revenue Recovery')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([2, 1])
    with sim_col1:
        custom_amount = st.number_input(
            "Transaction Amount (INR):",
            value=float(scen_meta["amount"]),
            step=500.0,
            help="You can test boundary amounts (e.g. ₹8,499 vs ₹8,501 for high-value escalation)."
        )
    with sim_col2:
        st.write("")
        st.write("")
        simulate_closed_loop = st.checkbox(
            "Simulate Closed-Loop Settlement (Customer Pays Link)",
            value=(current_scen_key == "auto_recovery"),
            help="If checked, automatically fires a mock Razorpay webhook settlement event upon successful action execution."
        )

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        run_sim = st.button("Simulate Payment Failure & Run Agent", type="primary", width='stretch')
    with btn_col2:
        reset_sim = st.button("Reset / Replay", width='stretch', help="Clears recent simulation state for a clean re-run")
        if reset_sim:
            st.session_state["latest_result"] = None
            st.session_state["latest_timeline"] = None
            st.rerun()

    if run_sim:
        with st.spinner("Ingesting failed payment and executing recovery decision pipeline..."):
            txn_id = f"demo_{current_scen_key[:4]}_{uuid.uuid4().hex[:6]}"
            
            # Build transaction dictionary
            if current_scen_key == "auto_recovery":
                txn = build_test_transaction(
                    transaction_id=txn_id,
                    customer_id="cust_buildathon_01",
                    merchant_id="merch_razorpay_demo",
                    amount=custom_amount,
                    customer_average_transaction=custom_amount,
                    failure_reason="network_timeout",
                    failure_category="transient",
                    customer_historical_success_rate=0.95,
                    customer_previous_transactions=12,
                    previous_failures_24h=0,
                    recovery_attempt_count=0,
                    ip_risk_score=0.04,
                    velocity_score=0.08
                )
            elif current_scen_key == "human_approval":
                txn = build_test_transaction(
                    transaction_id=txn_id,
                    customer_id="cust_enterprise_02",
                    merchant_id="merch_razorpay_demo",
                    amount=custom_amount,
                    customer_average_transaction=custom_amount,
                    failure_reason="network_timeout",
                    failure_category="transient",
                    customer_historical_success_rate=0.95,
                    customer_previous_transactions=15,
                    previous_failures_24h=0,
                    recovery_attempt_count=0,
                    ip_risk_score=0.05,
                    velocity_score=0.10
                )
            elif current_scen_key == "fraud_block":
                txn = build_test_transaction(
                    transaction_id=txn_id,
                    customer_id="cust_suspicious_03",
                    merchant_id="merch_razorpay_demo",
                    amount=custom_amount,
                    failure_reason="suspected_risk",
                    failure_category="risk_related",
                    ip_risk_score=0.92,
                    velocity_score=0.85,
                    customer_historical_success_rate=0.40,
                    previous_failures_24h=2
                )
            elif current_scen_key == "low_probability":
                txn = build_test_transaction(
                    transaction_id=txn_id,
                    customer_id="cust_churned_04",
                    merchant_id="merch_razorpay_demo",
                    amount=custom_amount,
                    customer_average_transaction=1200.0,
                    failure_reason="bank_declined",
                    failure_category="bank_decline",
                    payment_method="netbanking",
                    payment_network="axis",
                    payment_channel="web",
                    customer_historical_success_rate=0.0,
                    customer_previous_transactions=2,
                    previous_failures_24h=2,
                    recovery_attempt_count=2,
                    ip_risk_score=0.30,
                    velocity_score=0.85
                )
            else:
                txn = build_test_transaction(transaction_id=txn_id, amount=custom_amount)

            # Run LangGraph recovery agent
            res = run_agent(txn)

            # Persist to database
            db = SessionLocal()
            try:
                save_recovery_audit(db, res)
                
                # Optional closed loop settlement
                closed_loop_applied = False
                if simulate_closed_loop and res["policy_decision"] == "ACT":
                    p = db.query(FailedPayment).filter(FailedPayment.id == txn_id).first()
                    if p:
                        p.status = "RECOVERED"
                        db.commit()
                    res["verification_status"] = "SUCCESS"
                    res["money_recovered"] = custom_amount
                    res["agent_status"] = "RECOVERED"
                    closed_loop_applied = True

                timeline = construct_timeline(res, current_scen_key, closed_loop=closed_loop_applied)
                st.session_state["latest_result"] = res
                st.session_state["latest_timeline"] = timeline
            finally:
                db.close()

        # Deterministic verification check & Large Decision Result Badge
        act_decision = res.get("policy_decision", "REFUSE")
        is_matched = bool(act_decision == scen_meta.get("expected_decision"))

        if act_decision == "ACT":
            verdict_text = "ACT — AUTONOMOUS RECOVERY"
            verdict_color = "#15803D"
            verdict_bg = "#DCFCE7"
            verdict_border = "#86EFAC"
        elif act_decision == "ESCALATE":
            verdict_text = "ESCALATE — HUMAN REVIEW REQUIRED"
            verdict_color = "#B45309"
            verdict_bg = "#FEF3C7"
            verdict_border = "#FDE68A"
        else:
            verdict_text = "REFUSE — ACTION BLOCKED"
            verdict_color = "#B91C1C"
            verdict_bg = "#FEE2E2"
            verdict_border = "#FCA5A5"

        match_badge = '<span style="background:#DCFCE7; color:#15803D; border:1px solid #86EFAC; font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:9999px;">✓ Policy Match</span>' if is_matched else '<span style="background:#FEE2E2; color:#B91C1C; border:1px solid #FCA5A5; font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:9999px;">⚠ Policy Divergence</span>'

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid {verdict_color}; border-radius:10px; padding:20px; margin-top:18px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                    <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.04em;">Simulation Outcome</span>
                    <div style="font-size:1.15rem; font-weight:800; color:#0F172A; margin-top:2px;">
                        Transaction <code>{res['transaction_id']}</code>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    {match_badge}
                </div>
            </div>

            <div style="margin:14px 0 10px 0;">
                <span style="display:inline-block; font-size:1.05rem; font-weight:800; color:{verdict_color}; background:{verdict_bg}; border:1px solid {verdict_border}; padding:8px 18px; border-radius:8px; letter-spacing:0.02em;">
                    {verdict_text}
                </span>
            </div>

            <div style="font-size:0.92rem; color:#334155; line-height:1.5; margin-bottom:14px;">
                <strong>Policy Reason:</strong> {res.get('policy_reason', 'Evaluated against recovery policy rules.')}
            </div>

            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; font-size:0.85rem;">
                <div style="background:#F8FAFC; padding:10px 14px; border-radius:8px; border:1px solid #E2E8F0;">
                    <span style="color:#64748B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">Amount</span><br/>
                    <b style="color:#0F172A; font-size:1.15rem;">₹{res.get('amount', 0.0):,.2f}</b>
                </div>
                <div style="background:#F8FAFC; padding:10px 14px; border-radius:8px; border:1px solid #E2E8F0;">
                    <span style="color:#64748B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">Recovery Probability</span><br/>
                    <b style="color:#0F172A; font-size:1.15rem;">{res.get('recovery_probability', 0.0)*100:.1f}%</b>
                </div>
                <div style="background:#F8FAFC; padding:10px 14px; border-radius:8px; border:1px solid #E2E8F0;">
                    <span style="color:#64748B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">Expected Value (EV)</span><br/>
                    <b style="color:#0F172A; font-size:1.15rem;">₹{res.get('expected_recovery_value', 0.0):,.2f}</b>
                </div>
                <div style="background:#F8FAFC; padding:10px 14px; border-radius:8px; border:1px solid #E2E8F0;">
                    <span style="color:#64748B; font-size:0.75rem; font-weight:600; text-transform:uppercase;">Recommended Action</span><br/>
                    <b style="color:#2563EB; font-size:1.0rem;">{(res.get('recommended_action') or 'none').upper()}</b>
                </div>
            </div>

            <div style="margin-top:12px; padding:10px 14px; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0; font-size:0.85rem; color:#475569;">
                <b>{scen_meta.get('business_title', 'Business Impact')}:</b> {scen_meta.get('business_impact', 'Action completed according to deterministic policy rules.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render visual Agent Decision Timeline below simulation outcome
        render_agent_decision_timeline(res)

        if act_decision == "ESCALATE":
            st.info("Action Required: Transaction has been routed to Merchant Approval Queue. Switch to Tab 4 to review, approve, or reject.")
        else:
            st.info("Switch to Live Decision Trace (Tab 3) to view the deep JSON telemetry breakdown.")
    elif st.session_state.get("latest_result"):
        # Display existing simulation result and timeline if already processed in session
        render_agent_decision_timeline(st.session_state["latest_result"])


# =============================================================================
# TAB 3: LIVE AGENT DECISION TRACE (7-STAGE VISUAL PIPELINE)
# =============================================================================
with tab_trace:
    st.subheader("Live Decision Trace")
    st.caption("Real-time inspection of RecoverAI's multi-stage reasoning, prediction, deterministic safety, and webhook verification.")

    res = st.session_state.get("latest_result")
    timeline = st.session_state.get("latest_timeline")

    if not res or not timeline:
        st.info("No transaction has been simulated yet in this session. Click Simulate Payment Failure & Run Agent in the Demo Simulator tab to inspect the decision journey.")
    else:
        # Decision Banner
        decision = res.get("policy_decision", "REFUSE")
        amt = res.get("amount", 0.0)
        source_diag = res.get("diagnosis_source", "heuristic")
        source_rec = res.get("recommendation_source", "heuristic")

        if decision == "ACT":
            status_color = "#16A34A"
            status_bg = "#DCFCE7"
            status_border = "#86EFAC"
        elif decision == "ESCALATE":
            status_color = "#F59E0B"
            status_bg = "#FEF3C7"
            status_border = "#FDE68A"
        else:
            status_color = "#DC2626"
            status_bg = "#FEE2E2"
            status_border = "#FCA5A5"

        st.markdown(f"""
        <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-left: 5px solid {status_color}; padding:16px 20px; border-radius:8px; margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A; display:flex; justify-content:space-between; align-items:center;">
                <span>Transaction <code>{res.get('transaction_id')}</code></span>
                <span style="font-size:0.85rem; font-weight:700; color:{status_color}; background:{status_bg}; border:1px solid {status_border}; padding:3px 10px; border-radius:6px;">{decision}</span>
            </div>
            <div style="color:#475569; font-size:0.88rem; margin-top:6px;">
                Amount: <b>₹{amt:,.2f}</b> &nbsp;|&nbsp; 
                Failure Reason: <b>{res.get('failure_reason')}</b> &nbsp;|&nbsp; 
                ML Recovery Prob: <b>{res.get('recovery_probability', 0.0)*100:.1f}%</b> &nbsp;|&nbsp; 
                Agent Status: <b>{res.get('agent_status')}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent Decision Timeline at top of Tab 3
        render_agent_decision_timeline(res)

        st.markdown("---")
        st.markdown("#### Detailed Telemetry & Step Breakdown")

        # 7-STAGE PIPELINE CARDS
        for step in timeline:
            step_class = "step-card"
            if step['step_number'] in [5, 6]:
                if decision == "ACT":
                    step_class += " step-card-act"
                elif decision == "ESCALATE":
                    step_class += " step-card-escalate"
                else:
                    step_class += " step-card-refuse"

            with st.container():
                st.markdown(f"""
                <div class="{step_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.02rem; font-weight:700; color:#0F172A;">
                            Step {step['step_number']}: {step['title']}
                        </span>
                    </div>
                    <div style="color:#334155; font-size:0.90rem; margin-top:6px;">
                        {step['summary']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"Observable Telemetry & Metrics (Step {step['step_number']})", expanded=(step['step_number'] in [3, 4, 5])):
                    st.json(step["details"])

        # Manual Webhook Settlement Simulator
        if res.get("policy_decision") == "ACT" and res.get("agent_status") != "RECOVERED":
            st.markdown("---")
            st.markdown("### Live Webhook Simulator (Close Recovery Loop)")
            st.write("Simulate customer clicking the recovery payment link and completing payment via Razorpay.")
            if st.button("Simulate Customer Payment Webhook (Close Loop)", type="primary"):
                db = SessionLocal()
                try:
                    plink_id = f"plink_demo_{res['transaction_id'][-6:]}"
                    claim = db.query(PaymentExecutionClaim).filter(PaymentExecutionClaim.payment_id == res['transaction_id']).first()
                    if not claim:
                        create_execution_claim(
                            db=db,
                            idempotency_key=f"idemp_{res['transaction_id']}",
                            payment_id=res['transaction_id'],
                            action_type="payment_link",
                            amount=res['amount']
                        )
                        mark_execution_succeeded(
                            db=db,
                            idempotency_key=f"idemp_{res['transaction_id']}",
                            payment_link_id=plink_id,
                            short_url=f"https://rzp.io/i/{res['transaction_id'][-6:]}"
                        )
                    
                    mock_payload = {
                        "event": "payment_link.paid",
                        "payload": {
                            "payment_link": {
                                "entity": {
                                    "id": plink_id,
                                    "reference_id": res['transaction_id'],
                                    "status": "paid",
                                    "amount": int(res['amount'] * 100),
                                    "currency": "INR"
                                }
                            }
                        }
                    }
                    event = normalize_razorpay_webhook(mock_payload)
                    hook_res = process_webhook_lifecycle_event(db, event)
                    
                    # Update local state
                    res["verification_status"] = "SUCCESS"
                    res["money_recovered"] = res["amount"]
                    res["agent_status"] = "RECOVERED"
                    st.session_state["latest_result"] = res
                    st.success(f"Webhook processed successfully. Transaction marked RECOVERED (₹{res['amount']:,.2f}).")
                    st.rerun()
                finally:
                    db.close()


# =============================================================================
# TAB 4: MERCHANT APPROVAL QUEUE (HUMAN-IN-THE-LOOP GOVERNANCE)
# =============================================================================
with tab_approvals:
    st.subheader("Merchant Approval Queue (Human-in-the-Loop)")
    st.markdown("High-value payments and policy-escalated recovery actions are held in this queue until authorized by a merchant administrator.")

    db = SessionLocal()
    try:
        pending_items = list_pending_approvals(db)
    finally:
        db.close()

    if not pending_items:
        st.success("Approval queue is clear. No transactions currently require merchant review.")
        st.info("To test the human approval workflow, switch to the Demo Simulator tab and launch Scenario B (High-Value HITL Review).")
    else:
        st.warning(f"{len(pending_items)} high-value transaction(s) requiring merchant decision.")
        
        for item in pending_items:
            with st.container():
                st.markdown(f"""
                <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.05rem; font-weight:700; color:#0F172A;">
                            Transaction <code>{item.transaction_id}</code>
                        </span>
                        <span class="badge-escalate">PENDING APPROVAL</span>
                    </div>
                    <div style="margin-top:8px; color:#334155; font-size:0.92rem;">
                        <strong>Amount</strong>: <span style="color:#0F172A; font-weight:700;">₹{item.amount:,.2f} {item.currency}</span> &nbsp;|&nbsp;
                        <strong>Customer</strong>: <code>{item.customer_id}</code> &nbsp;|&nbsp;
                        <strong>Failure Reason</strong>: <code>{item.failure_reason}</code>
                    </div>
                    <div style="margin-top:6px; color:#64748B; font-size:0.85rem;">
                        <strong>ML Recovery Prob</strong>: <b>{item.recovery_probability*100:.1f}%</b> &nbsp;|&nbsp;
                        <strong>Recommended Action</strong>: <code>{item.recommended_action.upper()}</code> &nbsp;|&nbsp;
                        <strong>Policy Reason</strong>: {item.policy_reason}
                    </div>
                    <div style="margin-top:10px; background-color:#F8FAFC; border:1px solid #E2E8F0; padding:10px 12px; border-radius:6px; font-size:0.82rem; color:#475569;">
                        <strong>Decision Rationale</strong>: {getattr(item, 'recommendation_reason', None) or 'High expected value on reconnecting high-value account.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_col1, act_col2, _ = st.columns([1.5, 1.5, 4])
                
                with act_col1:
                    if st.button("Approve Recovery Action", key=f"app_{item.transaction_id}", type="primary"):
                        db = SessionLocal()
                        try:
                            ok, code, rec_res = approve_recovery_action(
                                db=db,
                                transaction_id=item.transaction_id,
                                human_notes="Authorized via Streamlit HITL Dashboard"
                            )
                            if ok:
                                st.success(f"Approved transaction `{item.transaction_id}`. Recovery action safely executed.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Approval failed: {rec_res.get('message', code)} ({code})")
                        finally:
                            db.close()

                with act_col2:
                    if st.button("Reject Recovery Action", key=f"rej_{item.transaction_id}"):
                        db = SessionLocal()
                        try:
                            ok, code, rej_res = reject_recovery_action(
                                db=db,
                                transaction_id=item.transaction_id,
                                human_notes="Rejected via Streamlit HITL Dashboard"
                            )
                            if ok:
                                st.warning(f"Transaction `{item.transaction_id}` rejected. Zero recovery actions executed.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Rejection failed: {rej_res.get('message', code)} ({code})")
                        finally:
                            db.close()

    st.markdown("---")
    st.caption("FinTech Safety Invariant: Fraud or risk-related blocked payments can never be approved or executed via this interface.")


# =============================================================================
# TAB 5: RECOVERY INSIGHTS & GOVERNANCE
# =============================================================================
with tab_insights:
    st.subheader("Recovery Insights, Calibration & Data Health")

    in_c1, in_c2 = st.columns(2)
    with in_c1:
        st.markdown("### Policy Threshold Comparison")
        st.write("Comparison between default industry threshold ($\\\\tau = 0.50$) vs. RecoverAI EV-Optimized threshold ($\\\\tau = 0.35$):")
        comp_df = pd.DataFrame([
            {"Policy": "Default Industry Standard (0.50)", "Recovered Revenue": f"₹{metrics['default_recovered']:,.2f}", "Action Cost": f"₹{metrics['default_cost']:,.2f}", "Net Profit": f"₹{metrics['default_net']:,.2f}"},
            {"Policy": "RecoverAI EV-Optimized (0.35)", "Recovered Revenue": f"₹{metrics['opt_recovered']:,.2f}", "Action Cost": f"₹{metrics['opt_cost']:,.2f}", "Net Profit": f"₹{metrics['opt_net']:,.2f}"},
            {"Policy": "Uplift Achieved by RecoverAI", "Recovered Revenue": f"+₹{metrics['gross_uplift_inr']:,.2f}", "Action Cost": f"+₹{metrics['inc_action_cost']:,.2f}", "Net Profit": f"+₹{metrics['net_uplift_inr']:,.2f}"}
        ])
        st.dataframe(comp_df, width='stretch', hide_index=True)

    with in_c2:
        st.markdown("### Population Stability Index (PSI) Drift Monitor")
        drift_report = run_drift_report(test_df, persist=False)
        st.metric("System Data Health Status", drift_report["system_status"].upper())
        st.write(f"Significant Drift Count: **{drift_report['significant_drift_count']}** | Moderate Drift Count: **{drift_report['moderate_drift_count']}**")
        with st.expander("View PSI Drift Feature Breakdown", expanded=False):
            st.dataframe(pd.DataFrame(drift_report["features"]), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### Immutable Audit Trail")
    st.caption("Decision records captured for traceability and governance. Source: `logs/decision_audit.jsonl`")

    audit_log_path = os.path.join(PROJECT_ROOT, "logs", "decision_audit.jsonl")
    if os.path.exists(audit_log_path):
        records = []
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except Exception:
                        pass
        
        if records:
            total_records = len(records)
            act_count = sum(1 for r in records if r.get("safety", {}).get("decision") == "ACT")
            esc_count = sum(1 for r in records if r.get("safety", {}).get("decision") == "ESCALATE")
            ref_count = sum(1 for r in records if r.get("safety", {}).get("decision") == "REFUSE")

            summary_cols = st.columns([1, 1, 1, 1])
            with summary_cols[0]:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Total Audit Records</div>
                    <div style="color:#0F172A; font-size:1.25rem; font-weight:700; margin-top:2px;">{total_records}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[1]:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #16A34A; border-radius:8px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Policy ACT</div>
                    <div style="color:#15803D; font-size:1.25rem; font-weight:700; margin-top:2px;">{act_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[2]:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #F59E0B; border-radius:8px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Escalated (HITL)</div>
                    <div style="color:#B45309; font-size:1.25rem; font-weight:700; margin-top:2px;">{esc_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[3]:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #DC2626; border-radius:8px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Policy Blocked</div>
                    <div style="color:#B91C1C; font-size:1.25rem; font-weight:700; margin-top:2px;">{ref_count}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            filter_col1, filter_col2 = st.columns([2, 3])
            with filter_col1:
                display_limit = st.selectbox(
                    "Records to display:",
                    options=[5, 10, 20, 50],
                    index=1,
                    key="audit_display_limit"
                )
            with filter_col2:
                decision_filter = st.selectbox(
                    "Filter by Policy Outcome:",
                    options=["ALL", "ACT", "ESCALATE", "REFUSE"],
                    index=0,
                    key="audit_decision_filter"
                )

            filtered_records = records
            if decision_filter != "ALL":
                filtered_records = [r for r in records if r.get("safety", {}).get("decision") == decision_filter]

            visible_records = filtered_records[:display_limit]
            st.caption(f"Displaying {len(visible_records)} of {len(filtered_records)} matching audit records (ordered chronologically, newest first):")

            for record in visible_records:
                txn_id = record.get("transaction_id", "N/A")
                ts = record.get("timestamp", "N/A")
                cust_id = record.get("customer_id", "N/A")
                amt = record.get("amount", 0.0)

                pred = record.get("prediction", {}) or {}
                prob = pred.get("recovery_probability", 0.0)
                model_type = pred.get("model_type", "EXP_0 Logistic Regression")
                exp_id = pred.get("experiment_id", "EXP_0")
                explanation = pred.get("explanation", "Prediction generated by recovery model.")
                pos_signals = pred.get("top_positive_signals", []) or []
                neg_signals = pred.get("top_negative_signals", []) or []

                bp = record.get("business_policy", {}) or {}
                tau = bp.get("threshold", 0.35)
                rec_action = bp.get("recommended_action", "none")
                action_cost = bp.get("action_cost", 0.0)
                gross_ev = bp.get("expected_gross_recovery_value", 0.0)

                safety = record.get("safety", {}) or {}
                decision = safety.get("decision", "REFUSE")
                justification = safety.get("justification", "Evaluated against deterministic policy guardrails.")
                triggered_rules = safety.get("triggered_rules", [])

                execution = record.get("execution", {}) or {}
                exec_status = execution.get("execution_status", "NOT_EXECUTED")
                dry_run = execution.get("dry_run", True)
                ref_id = execution.get("reference_id") or "N/A"
                blocking_reason = execution.get("blocking_reason") or "None"

                if decision == "ACT":
                    status_color = "#16A34A"
                    status_bg = "#DCFCE7"
                    status_border = "#86EFAC"
                elif decision == "ESCALATE":
                    status_color = "#F59E0B"
                    status_bg = "#FEF3C7"
                    status_border = "#FDE68A"
                else:
                    status_color = "#DC2626"
                    status_bg = "#FEE2E2"
                    status_border = "#FCA5A5"

                with st.container():
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {status_color}; border-radius:8px; padding:12px 16px; margin-bottom:8px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                            <div>
                                <span style="font-weight:700; color:#0F172A; font-size:0.95rem;">Transaction <code>{txn_id}</code></span>
                                <span style="color:#64748B; font-size:0.80rem; margin-left:12px;">{ts}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:0.80rem; font-weight:700; color:{status_color}; background:{status_bg}; padding:2px 8px; border:1px solid {status_border}; border-radius:4px;">{decision}</span>
                                <span style="font-size:0.75rem; color:#475569; background:#F1F5F9; border:1px solid #E2E8F0; padding:2px 6px; border-radius:4px;">{exec_status}</span>
                            </div>
                        </div>
                        <div style="margin-top:8px; display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-size:0.82rem;">
                            <div><span style="color:#64748B;">Customer:</span> <code>{cust_id}</code></div>
                            <div><span style="color:#64748B;">Amount:</span> <b style="color:#0F172A;">₹{amt:,.2f}</b></div>
                            <div><span style="color:#64748B;">P(Recovery):</span> <b style="color:#0F172A;">{prob*100:.1f}%</b></div>
                            <div><span style="color:#64748B;">Recommended:</span> <code>{rec_action.upper()}</code></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Inspect Record — {txn_id}", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:6px;">
                                Prediction &amp; Model Evidence
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px; font-size:0.80rem; color:#334155;">
                                <div><span style="color:#64748B;">Model:</span> <b style="color:#0F172A;">{model_type} ({exp_id})</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Calibrated Probability:</span> <b style="color:#0F172A;">{prob*100:.2f}%</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Expected Gross Value:</span> <b style="color:#0F172A;">₹{gross_ev:,.2f}</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Decision Cutoff (&tau;):</span> <code>{tau}</code></div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c2:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:6px;">
                                Policy Guard Verdict
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px; font-size:0.80rem; color:#334155;">
                                <div><span style="color:#64748B;">Decision:</span> <b style="color:{status_color};">{decision}</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Action Cost:</span> <b style="color:#0F172A;">₹{action_cost:,.2f}</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Triggered Rules:</span> <code>{', '.join(triggered_rules) if triggered_rules else 'None (Clean Pass)'}</code></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Authority:</span> <b style="color:#0F172A;">Deterministic Policy Engine</b></div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c3:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:6px;">
                                Execution &amp; Telemetry State
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px; font-size:0.80rem; color:#334155;">
                                <div><span style="color:#64748B;">Status:</span> <b style="color:#0F172A;">{exec_status}</b></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Execution Mode:</span> <code>{'Dry Run' if dry_run else 'Live Test'}</code></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Reference ID:</span> <code>{ref_id}</code></div>
                                <div style="margin-top:4px;"><span style="color:#64748B;">Blocking Reason:</span> <span>{blocking_reason}</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="margin-top:8px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#334155;">
                            <div style="color:#64748B; font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:4px;">Policy Justification</div>
                            {justification}
                        </div>
                        """, unsafe_allow_html=True)

                        if pos_signals or neg_signals:
                            sig_col1, sig_col2 = st.columns(2)
                            with sig_col1:
                                if pos_signals:
                                    st.markdown("""
                                    <div style="font-size:0.75rem; font-weight:600; color:#15803D; margin-top:6px; margin-bottom:4px;">
                                        Top Positive Signals
                                    </div>
                                    """, unsafe_allow_html=True)
                                    pos_df = pd.DataFrame(pos_signals)
                                    st.dataframe(pos_df, width='stretch', hide_index=True)
                            with sig_col2:
                                if neg_signals:
                                    st.markdown("""
                                    <div style="font-size:0.75rem; font-weight:600; color:#B91C1C; margin-top:6px; margin-bottom:4px;">
                                        Top Negative Signals
                                    </div>
                                    """, unsafe_allow_html=True)
                                    neg_df = pd.DataFrame(neg_signals)
                                    st.dataframe(neg_df, width='stretch', hide_index=True)

                        with st.expander("View Raw Audit Record (JSON)", expanded=False):
                            st.json(record)
                    
                    st.write("")
        else:
            st.write("Audit log is currently empty.")
    else:
        st.write("No audit log found on disk.")
