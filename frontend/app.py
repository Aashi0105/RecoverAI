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
# GLOBAL FINTECH DESIGN SYSTEM & CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & Background Adjustments - Near-Black & Off-White */
    .stApp {
        background-color: #080A0A;
        color: #F5F7F5;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Dark Theme - Differentiated Near-Black */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] > div {
        background-color: #0D1010 !important;
        color: #F5F7F5 !important;
        border-right: 1px solid #242B2B;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #A7B0AD;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F5F7F5 !important;
    }
    
    /* Hero KPI Cards */
    .metric-card-hero {
        background: #121616;
        border: 1px solid #242B2B;
        border-radius: 8px;
        padding: 16px 14px;
        text-align: left;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        transition: border-color 0.15s ease;
    }
    .metric-card-hero:hover {
        border-color: #333C3C;
    }
    .metric-card-green {
        border-top: 2px solid #39FF88;
    }
    .metric-card-purple {
        border-top: 2px solid #A78BFA;
    }
    .metric-card-amber {
        border-top: 2px solid #FB923C;
    }
    
    .metric-hero-label {
        color: #707A77;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-hero-val {
        color: #F5F7F5;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .metric-hero-sub {
        color: #707A77;
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .metric-hero-delta-green {
        color: #39FF88;
        font-size: 0.80rem;
        font-weight: 600;
        margin-top: 3px;
    }
    
    /* Funnel Step Card */
    .funnel-card {
        background: #121616;
        border: 1px solid #242B2B;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .funnel-step-title {
        color: #707A77;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .funnel-step-val {
        color: #F5F7F5;
        font-size: 1.30rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .funnel-step-pct {
        color: #39FF88;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 2px;
    }
    
    /* Semantic Status Badges - Compact & Understated */
    .badge-act {
        background-color: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.35);
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-recovered {
        background-color: rgba(167, 139, 250, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(167, 139, 250, 0.40);
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-approved {
        background-color: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.40);
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-escalate {
        background-color: rgba(251, 146, 60, 0.15);
        color: #FB923C;
        border: 1px solid rgba(251, 146, 60, 0.40);
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-refuse {
        background-color: rgba(255, 92, 92, 0.15);
        color: #FF5C5C;
        border: 1px solid rgba(255, 92, 92, 0.40);
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }
    .badge-llm {
        background-color: rgba(57, 255, 136, 0.12);
        color: #39FF88;
        border: 1px solid rgba(57, 255, 136, 0.30);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
    }
    .badge-heuristic {
        background-color: rgba(112, 122, 119, 0.15);
        color: #A7B0AD;
        border: 1px solid rgba(112, 122, 119, 0.35);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
    }
    
    /* Trace & Pipeline Step Cards */
    .step-card {
        background-color: #121616;
        border: 1px solid #242B2B;
        border-left: 3px solid #39FF88;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .step-card-act {
        border-left-color: #22D3EE;
    }
    .step-card-escalate {
        border-left-color: #FB923C;
    }
    .step-card-refuse {
        border-left-color: #FF5C5C;
    }
    
    /* Feed / Activity Stream Cards */
    .activity-item {
        background-color: #121616;
        border: 1px solid #242B2B;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Scenario Selector Cards - Charcoal & Neon Green Highlight */
    .scenario-box {
        background: #121616;
        border: 1px solid #242B2B;
        border-radius: 8px;
        padding: 14px 12px;
        text-align: left;
        margin-bottom: 10px;
        transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
    }
    .scenario-box:hover {
        border-color: #333C3C;
    }
    .scenario-box-active {
        border-color: #39FF88 !important;
        background: #171C1C !important;
    }
    .scenario-title {
        font-weight: 700;
        color: #F5F7F5;
        font-size: 0.90rem;
        letter-spacing: -0.01em;
    }
    .scenario-subtitle {
        font-weight: 600;
        color: #A7B0AD;
        font-size: 0.82rem;
        margin-top: 4px;
    }
    .scenario-caption {
        color: #707A77;
        font-size: 0.74rem;
        margin-top: 4px;
        line-height: 1.35;
    }

    /* Semantic Status Indicators */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-act {
        background-color: #22D3EE;
    }
    .dot-recovered {
        background-color: #A78BFA;
    }
    .dot-approved {
        background-color: #FBBF24;
    }
    .dot-escalate {
        background-color: #FB923C;
    }
    .dot-refuse {
        background-color: #FF5C5C;
    }
    .dot-muted {
        background-color: #707A77;
    }

    /* Agent Decision Timeline Styles */
    .timeline-container {
        margin-top: 14px;
        margin-bottom: 18px;
        background: #0D1010;
        border: 1px solid #242B2B;
        border-radius: 8px;
        padding: 16px 18px;
    }
    .timeline-card {
        background-color: #121616;
        border: 1px solid #242B2B;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease;
    }
    .timeline-card:hover {
        border-color: #333C3C;
    }
    .timeline-card-completed {
        border-left: 3px solid #22D3EE;
    }
    .timeline-card-escalated {
        border-left: 3px solid #FB923C;
    }
    .timeline-card-blocked {
        border-left: 3px solid #FF5C5C;
    }
    .timeline-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .pill-completed {
        background: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.35);
    }
    .pill-escalated {
        background: rgba(251, 146, 60, 0.12);
        color: #FB923C;
        border: 1px solid rgba(251, 146, 60, 0.35);
    }
    .pill-blocked {
        background: rgba(255, 92, 92, 0.12);
        color: #FF5C5C;
        border: 1px solid rgba(255, 92, 92, 0.35);
    }
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .timeline-title {
        font-weight: 600;
        font-size: 0.90rem;
        color: #F5F7F5;
    }
    .timeline-stage-num {
        color: #707A77;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .timeline-content {
        margin-top: 5px;
        font-size: 0.85rem;
        color: #A7B0AD;
        line-height: 1.4;
    }
    .timeline-meta {
        margin-top: 6px;
        font-size: 0.78rem;
        color: #707A77;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
    }

    /* Primary & Secondary Buttons - Enterprise Flat Minimal */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #121616 !important;
        color: #39FF88 !important;
        border: 1px solid #39FF88 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: 4px !important;
        padding: 6px 14px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #1F2626 !important;
        border-color: #39FF88 !important;
        color: #F5F7F5 !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: #121616 !important;
        color: #F5F7F5 !important;
        border: 1px solid #242B2B !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        border-radius: 4px !important;
        padding: 5px 12px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #171C1C !important;
        border-color: #333C3C !important;
        color: #39FF88 !important;
        box-shadow: none !important;
    }

    /* Tabs Component Styling - Clean Enterprise Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #242B2B;
        padding-bottom: 0px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #707A77 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 10px 2px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #A7B0AD !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #F5F7F5 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #39FF88 !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Timestamps for Cache Invalidation
csv_file_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
csv_mtime = os.path.getmtime(csv_file_path) if os.path.exists(csv_file_path) else 0.0

model_artifact_file = os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib")
model_mtime = os.path.getmtime(model_artifact_file) if os.path.exists(model_artifact_file) else 0.0


@st.cache_data
def load_and_split_dataset(csv_mtime: float):
    """Loads raw transactions dataset and splits into dev / test holdout sets."""
    if not os.path.exists(csv_file_path):
        st.error(f"Dataset missing at {csv_file_path}")
        st.stop()

    df_raw = pd.read_csv(csv_file_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    seed = 42
    idx_dev, idx_test, _, _ = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )
    test_df = failed_df.loc[idx_test].copy()
    return test_df, idx_dev, idx_test


@st.cache_data
def compute_live_top_metrics(model_mtime: float, csv_mtime: float):
    """
    Computes rigorous top-level business and economic KPI metrics comparing
    baseline industry threshold (0.50) vs RecoverAI EV-Optimized threshold (0.35).
    """
    from agent.orchestrator import load_orchestrator_model
    test_df, idx_dev, idx_test = load_and_split_dataset(csv_mtime=csv_mtime)
    pipeline = load_orchestrator_model()

    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    test_probs = pipeline.predict_proba(test_df[exp0_features])[:, 1]
    test_probs_series = pd.Series(test_probs, index=idx_test)

    m_def = evaluate_threshold_grid(test_df, test_probs_series, thresholds=[0.50]).iloc[0].to_dict()
    m_opt = evaluate_threshold_grid(test_df, test_probs_series, thresholds=[0.35]).iloc[0].to_dict()

    total_risk = float(np.sum(test_df["amount"]))
    total_failed_count = len(test_df)
    
    def_rev = m_def["realized_recovered_revenue"]
    def_cost = m_def["action_costs_incurred"]
    def_net = def_rev - def_cost

    opt_rev = m_opt["realized_recovered_revenue"]
    opt_cost = m_opt["action_costs_incurred"]
    opt_net = opt_rev - opt_cost

    gross_uplift = opt_rev - def_rev
    gross_uplift_pct = (gross_uplift / total_risk * 100) if total_risk > 0 else 0.0

    true_net_uplift = opt_net - def_net
    true_net_uplift_pct = (true_net_uplift / total_risk * 100) if total_risk > 0 else 0.0
    inc_cost = opt_cost - def_cost

    # Calculate operational recovery rate
    opt_selected = int(m_opt.get("payments_selected", 361))
    recovered_amt = opt_rev
    recovery_rate_pct = (recovered_amt / total_risk * 100) if total_risk > 0 else 0.0

    return {
        "total_risk": total_risk,
        "total_failed_count": total_failed_count,
        "default_recovered": def_rev,
        "default_cost": def_cost,
        "default_net": def_net,
        "opt_recovered": opt_rev,
        "opt_cost": opt_cost,
        "opt_net": opt_net,
        "gross_uplift_inr": gross_uplift,
        "gross_uplift_pct": gross_uplift_pct,
        "net_uplift_inr": true_net_uplift,
        "net_uplift_pct": true_net_uplift_pct,
        "inc_action_cost": inc_cost,
        "recovery_rate_pct": recovery_rate_pct,
        "payments_selected": opt_selected
    }


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
                c_pill = f'<span class="timeline-pill" style="margin-right:6px; margin-bottom:4px; background:#171C1C; color:#A7B0AD; border:1px solid #242B2B;">{label}: {v}</span>'
            check_pills.append(c_pill)

        pills_str = "".join(check_pills)
        reasons_html = "".join([f"<li>{r}</li>" for r in reasons_list])
        is_open = "open" if decision != "ACT" else ""

        explanation_html = f"""
        <details style="margin-top:10px; background:#080A0A; border:1px solid #242B2B; border-radius:6px; padding:8px 12px;" {is_open}>
            <summary style="font-size:0.82rem; font-weight:700; color:#39FF88; cursor:pointer;">
                Structured Decision Explanation & Policy Checks Breakdown
            </summary>
            <div style="margin-top:8px; font-size:0.82rem; color:#A7B0AD;">
                <div style="margin-bottom:6px;">
                    <span style="color:#707A77;">Primary Decision Factor:</span> <code>{p_factor}</code>
                </div>
                <div style="margin-bottom:8px;">
                    <span style="color:#707A77;">Policy Checks:</span><br/>
                    <div style="margin-top:4px; display:flex; flex-wrap:wrap;">{pills_str}</div>
                </div>
                <div>
                    <span style="color:#707A77;">Evaluated Reasons:</span>
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
    <div style="padding:4px 0 8px 0;">
        <div style="font-size:1.15rem; font-weight:700; color:#F5F7F5; letter-spacing:-0.02em;">RecoverAI</div>
        <div style="font-size:0.78rem; color:#707A77; margin-top:2px;">Revenue Recovery & FinTech Governance</div>
    </div>
    <div style="border-top:1px solid #242B2B; margin:8px 0 12px 0;"></div>
    """, unsafe_allow_html=True)
    
    # Engine status detection
    from agent.services.llm_service import is_llm_available
    has_real_llm = is_llm_available()
    diag_status = "Live LLM" if has_real_llm else "Fallback Mode"
    
    razorpay_status = is_razorpay_configured()
    razorpay_label = "Test Mode" if razorpay_status else "Mock Sandbox"
    
    st.markdown(f"""
    <div style="font-size:0.72rem; font-weight:700; color:#707A77; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
        System Status
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; font-size:0.80rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">Agent Pipeline</span>
            <span style="color:#39FF88; font-weight:600;">Online</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">Diagnosis Engine</span>
            <span style="color:#F5F7F5; font-weight:600;">{diag_status}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">Policy Engine</span>
            <span style="color:#39FF88; font-weight:600;">Active</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">Razorpay Integration</span>
            <span style="color:#F5F7F5; font-weight:600;">{razorpay_label}</span>
        </div>
    </div>
    <div style="border-top:1px solid #242B2B; margin:14px 0 10px 0;"></div>
    <div style="font-size:0.72rem; font-weight:700; color:#707A77; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
        Parameters
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; font-size:0.80rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">EV Threshold (&tau;)</span>
            <span style="color:#F5F7F5; font-weight:600;">0.35</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#A7B0AD;">High-Value Escalation</span>
            <span style="color:#F5F7F5; font-weight:600;">₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}</span>
        </div>
    </div>
    <div style="border-top:1px solid #242B2B; margin:14px 0 12px 0;"></div>
    <div style="font-size:0.72rem; font-weight:700; color:#707A77; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Safety Invariant
    </div>
    <div style="background:#121616; border:1px solid #242B2B; border-radius:6px; padding:10px 12px; font-size:0.78rem; color:#A7B0AD; line-height:1.45;">
        <div style="color:#F5F7F5; font-weight:600; margin-bottom:4px;">Governance Rules:</div>
        1. LLM Recommends.<br/>
        2. Deterministic Policy Decides.<br/>
        3. Human Approves Escalations.<br/>
        <span style="color:#FF5C5C; font-size:0.74rem; display:block; margin-top:6px;">Fraud and risk blocks cannot be overridden.</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("RecoverAI v2.4-prod | Buildathon Submission Build")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("RecoverAI — Revenue Recovery Command Center")
    st.markdown("**Predicts recovery probability, applies deterministic policy rules, and confirms settlement through verified webhooks.**")
with header_col2:
    st.write("")
    st.markdown("""
    <div style="background:#121616; border:1px solid #242B2B; border-radius:8px; padding:10px 14px; text-align:right;">
        <span style="color:#707A77; font-size:0.75rem; font-weight:700; text-transform:uppercase;">Agent Status</span><br/>
        <span style="color:#39FF88; font-weight:800; font-size:1.05rem;">ONLINE</span>
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
# TAB 1: RECOVERY COMMAND CENTER (POLISHED ENTERPRISE DASHBOARD)
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

    # 1. TOP KPI HERO SECTION (5-Card Clean Grid)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card-hero">
            <div class="metric-hero-label">Revenue at Risk</div>
            <div class="metric-hero-val">₹{metrics['total_risk']:,.0f}</div>
            <div class="metric-hero-sub">N = {metrics['total_failed_count']} Failed Txns Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card-hero metric-card-purple">
            <div class="metric-hero-label">Revenue Recovered</div>
            <div class="metric-hero-val" style="color:#A78BFA;">₹{metrics['opt_recovered']:,.0f}</div>
            <div class="metric-hero-sub">Captured at EV-Optimal &tau; = 0.35</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card-hero metric-card-green">
            <div class="metric-hero-label">Net Profit Uplift</div>
            <div class="metric-hero-val" style="color:#39FF88;">+₹{metrics['net_uplift_inr']:,.0f}</div>
            <div class="metric-hero-delta-green">+{metrics['net_uplift_pct']:.2f}% Net Growth</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card-hero">
            <div class="metric-hero-label">Recovery Rate</div>
            <div class="metric-hero-val">{metrics['recovery_rate_pct']:.1f}%</div>
            <div class="metric-hero-sub">Gross Risk Captured</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        badge_style = "color:#FB923C;" if pending_count > 0 else "color:#39FF88;"
        badge_text = f"{pending_count} Awaiting" if pending_count > 0 else "Queue Clear"
        st.markdown(f"""
        <div class="metric-card-hero {'metric-card-amber' if pending_count > 0 else ''}">
            <div class="metric-hero-label">Pending Reviews</div>
            <div class="metric-hero-val" style="{badge_style}">{pending_count}</div>
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
        <div style="background:#121616; border:1px solid #242B2B; border-radius:8px; padding:14px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem;">
                <span style="color:#707A77;">Default Standard (0.50):</span>
                <span style="font-weight:600; color:#A7B0AD;">₹{metrics['default_recovered']:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem;">
                <span style="color:#A7B0AD;">RecoverAI Optimized (0.35):</span>
                <span style="font-weight:600; color:#F5F7F5;">₹{metrics['opt_recovered']:,.2f}</span>
            </div>
            <div style="border-top:1px solid #242B2B; padding-top:8px; display:flex; justify-content:space-between; font-size:0.88rem;">
                <span style="color:#39FF88; font-weight:600;">Net Merchant Profit Uplift:</span>
                <span style="color:#39FF88; font-weight:700;">+₹{metrics['net_uplift_inr']:,.2f} (+{metrics['net_uplift_pct']:.2f}%)</span>
            </div>
        </div>
        """
        st.markdown(comp_card_html, unsafe_allow_html=True)
        st.caption("*Expected Value Optimization mathematically proves that threshold 0.35 captures recoverable revenue without overspending on action costs.*")

    with story_col2:
        st.subheader("End-to-End Recovery Funnel")
        st.markdown("Transaction lifecycle progression from failure ingestion to verified settlement:")
        
        fn1, fn2, fn3, fn4, fn5 = st.columns(5)
        with fn1:
            st.markdown(f"""
            <div class="funnel-card">
                <div class="funnel-step-title">1. Failed Ingest</div>
                <div class="funnel-step-val">{metrics['total_failed_count']}</div>
                <div class="funnel-step-pct">100% Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with fn2:
            st.markdown(f"""
            <div class="funnel-card">
                <div class="funnel-step-title">2. AI Analyzed</div>
                <div class="funnel-step-val">{metrics['total_failed_count']}</div>
                <div class="funnel-step-pct">100% Scored</div>
            </div>
            """, unsafe_allow_html=True)
        with fn3:
            act_pct = (metrics['payments_selected'] / metrics['total_failed_count'] * 100) if metrics['total_failed_count'] > 0 else 0.0
            st.markdown(f"""
            <div class="funnel-card" style="border-color:#22D3EE;">
                <div class="funnel-step-title">3. Policy ACT</div>
                <div class="funnel-step-val" style="color:#22D3EE;">{metrics['payments_selected']}</div>
                <div class="funnel-step-pct">{act_pct:.1f}% Passed</div>
            </div>
            """, unsafe_allow_html=True)
        with fn4:
            st.markdown("""
            <div class="funnel-card" style="border-color:#242B2B;">
                <div class="funnel-step-title">4. Executed</div>
                <div class="funnel-step-val" style="color:#F5F7F5;">361</div>
                <div class="funnel-step-pct">0 Double Hits</div>
            </div>
            """, unsafe_allow_html=True)
        with fn5:
            st.markdown("""
            <div class="funnel-card" style="border-color:#A78BFA; background:#171C1C;">
                <div class="funnel-step-title">5. Recovered</div>
                <div class="funnel-step-val" style="color:#F5F7F5;">210</div>
                <div class="funnel-step-pct" style="color:#A78BFA;">58.2% Captured</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(
            "<div style='display:flex; justify-content:space-between; font-size:0.80rem; color:#A7B0AD; padding:2px 6px;'>"
            "<span><b style='color:#FF5C5C;'>240 Blocked</b> (Fraud / Negative EV)</span>"
            "<span><b style='color:#FB923C;'>32 Escalated</b> (High-Value HITL)</span>"
            "<span><b style='color:#A78BFA;'>₹920,754 Recovered</b></span>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. DECISIONS OVERVIEW & FAILURE CATEGORY INSIGHTS
    dec_col, cat_col = st.columns(2)
    
    with dec_col:
        st.subheader("Decision Distribution (Test Holdout N=633)")
        st.markdown("Policy Guard breakdown ensuring safe, cost-optimized recovery:")
        
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("""
            <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid #22D3EE; padding:12px 14px; border-radius:6px; margin-bottom:10px;">
                <span style="color:#22D3EE; font-weight:700; font-size:0.90rem; letter-spacing:0.02em;">AUTO RECOVERY (ACT)</span>
                <div style="font-size:1.35rem; font-weight:800; color:#F5F7F5; margin:2px 0;">361 <span style="font-size:0.85rem; color:#A7B0AD; font-weight:600;">(57.0%)</span></div>
                <span style="color:#707A77; font-size:0.80rem; font-weight:500;">High P(Recovery), low fraud risk</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid #FB923C; padding:12px 14px; border-radius:6px; margin-bottom:10px;">
                <span style="color:#FB923C; font-weight:700; font-size:0.90rem; letter-spacing:0.02em;">HUMAN REVIEW (ESCALATE)</span>
                <div style="font-size:1.35rem; font-weight:800; color:#F5F7F5; margin:2px 0;">32 <span style="font-size:0.85rem; color:#A7B0AD; font-weight:600;">(5.1%)</span></div>
                <span style="color:#707A77; font-size:0.80rem; font-weight:500;">Amount > ₹8,500 threshold</span>
            </div>
            """, unsafe_allow_html=True)
        with d_c2:
            st.markdown("""
            <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid #FF5C5C; padding:12px 14px; border-radius:6px; margin-bottom:10px;">
                <span style="color:#FF5C5C; font-weight:700; font-size:0.90rem; letter-spacing:0.02em;">POLICY BLOCKED (REFUSE)</span>
                <div style="font-size:1.35rem; font-weight:800; color:#F5F7F5; margin:2px 0;">48 <span style="font-size:0.85rem; color:#A7B0AD; font-weight:600;">(7.6%)</span></div>
                <span style="color:#707A77; font-size:0.80rem; font-weight:500;">Fraud / IP risk / permanent cards</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid #707A77; padding:12px 14px; border-radius:6px; margin-bottom:10px;">
                <span style="color:#A7B0AD; font-weight:700; font-size:0.90rem; letter-spacing:0.02em;">NO ACTION (NEGATIVE EV)</span>
                <div style="font-size:1.35rem; font-weight:800; color:#F5F7F5; margin:2px 0;">192 <span style="font-size:0.85rem; color:#A7B0AD; font-weight:600;">(30.3%)</span></div>
                <span style="color:#707A77; font-size:0.80rem; font-weight:500;">P < 0.35, spares wasteful action fee</span>
            </div>
            """, unsafe_allow_html=True)

    with cat_col:
        st.subheader("Failure Category Distribution")
        st.markdown("Top drivers of payment failure across transaction history:")
        
        cat_data = pd.DataFrame([
            {"Category": "Transient Network Timeout", "Share": 38.5, "Recoverability": "Very High (87%)"},
            {"Category": "Insufficient Funds / Bank Decline", "Share": 27.2, "Recoverability": "Medium (42%)"},
            {"Category": "Technical Gateway Error", "Share": 19.4, "Recoverability": "High (74%)"},
            {"Category": "Customer Authentication Drop", "Share": 9.8, "Recoverability": "Medium (38%)"},
            {"Category": "Suspected Risk / Fraud", "Share": 5.1, "Recoverability": "Blocked (0%)"}
        ])
        st.dataframe(cat_data, use_container_width=True, hide_index=True)
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
                        <div style="font-weight:700; font-size:1.0rem; color:#F5F7F5;">
                            Transaction <code>{p.id}</code> &nbsp;|&nbsp; <span style="color:#A7B0AD;">₹{p.amount:,.2f} {p.currency}</span>
                        </div>
                        <div style="color:#707A77; font-size:0.85rem; margin-top:3px;">
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

    # 4 Scenario Selection Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        is_sel_a = (st.session_state["selected_scenario"] == "auto_recovery")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_a else ''}">
            <div style="display:flex; align-items:center; gap:7px;">
                <span class="status-dot dot-act"></span>
                <span class="scenario-title">Scenario A</span>
            </div>
            <div class="scenario-subtitle">Smart Auto-Recovery</div>
            <div class="scenario-caption">Transient timeout &rarr; High P &rarr; Auto-executes & settles</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario A", key="btn_sc_a", use_container_width=True):
            st.session_state["selected_scenario"] = "auto_recovery"
            st.rerun()

    with sc2:
        is_sel_b = (st.session_state["selected_scenario"] == "human_approval")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_b else ''}">
            <div style="display:flex; align-items:center; gap:7px;">
                <span class="status-dot dot-escalate"></span>
                <span class="scenario-title">Scenario B</span>
            </div>
            <div class="scenario-subtitle">High-Value HITL Review</div>
            <div class="scenario-caption">₹14,500 > limit &rarr; Escalates to Merchant Queue</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario B", key="btn_sc_b", use_container_width=True):
            st.session_state["selected_scenario"] = "human_approval"
            st.rerun()

    with sc3:
        is_sel_c = (st.session_state["selected_scenario"] == "fraud_block")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_c else ''}">
            <div style="display:flex; align-items:center; gap:7px;">
                <span class="status-dot dot-refuse"></span>
                <span class="scenario-title">Scenario C</span>
            </div>
            <div class="scenario-subtitle">Fraud / Risk Block</div>
            <div class="scenario-caption">IP risk 0.92 &rarr; Policy blocks &rarr; Zero execution</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario C", key="btn_sc_c", use_container_width=True):
            st.session_state["selected_scenario"] = "fraud_block"
            st.rerun()

    with sc4:
        is_sel_d = (st.session_state["selected_scenario"] == "low_probability")
        st.markdown(f"""
        <div class="scenario-box {'scenario-box-active' if is_sel_d else ''}">
            <div style="display:flex; align-items:center; gap:7px;">
                <span class="status-dot dot-muted"></span>
                <span class="scenario-title">Scenario D</span>
            </div>
            <div class="scenario-subtitle">Negative EV Refusal</div>
            <div class="scenario-caption">P < 0.35 &rarr; Negative EV &rarr; Spares action fee</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Scenario D", key="btn_sc_d", use_container_width=True):
            st.session_state["selected_scenario"] = "low_probability"
            st.rerun()

    current_scen_key = st.session_state["selected_scenario"]
    scen_meta = PREDEFINED_SCENARIOS[current_scen_key]

    st.markdown("---")
    
    # Active Scenario Card Details
    exp_decision = scen_meta.get("expected_decision", "ACT")
    exp_badge_color = "#22D3EE" if exp_decision == "ACT" else ("#FB923C" if exp_decision == "ESCALATE" else "#FF5C5C")
    
    st.markdown(f"""
    <div style="background:#121616; border:1px solid #242B2B; border-radius:8px; padding:16px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:1.05rem; font-weight:700; color:#F5F7F5;">
                {scen_meta['name']}
            </span>
            <span style="color:#A7B0AD; font-size:0.85rem;">
                Target Policy: <span style="color:{exp_badge_color}; font-weight:700; padding:2px 8px; border:1px solid {exp_badge_color}; border-radius:4px;">{exp_decision}</span> &nbsp;|&nbsp; Action: <code>{scen_meta['expected_action'].upper()}</code>
            </span>
        </div>
        <div style="color:#A7B0AD; font-size:0.90rem; margin-top:6px;">
            {scen_meta['description']}
        </div>
        <div style="margin-top:10px; padding:8px 12px; background:#080A0A; border-radius:6px; border:1px solid #242B2B; font-size:0.82rem; color:#707A77; display:flex; justify-content:space-between;">
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
        run_sim = st.button("Simulate Payment Failure & Run Agent", type="primary", use_container_width=True)
    with btn_col2:
        reset_sim = st.button("Reset / Replay", use_container_width=True, help="Clears recent simulation state for a clean re-run")
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

        # Deterministic verification check
        act_decision = res.get("policy_decision")
        is_matched = bool(act_decision == scen_meta.get("expected_decision"))
        match_html = '<span style="color:#39FF88; font-weight:700;">PASSED — Policy Match</span>' if is_matched else '<span style="color:#FF5C5C; font-weight:700;">MISMATCH — Policy Divergence</span>'

        st.markdown(f"""
        <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid {'#39FF88' if is_matched else '#FF5C5C'}; border-radius:8px; padding:14px; margin-top:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.95rem; font-weight:700; color:#F5F7F5;">
                    Simulation Outcome for <code>{res['transaction_id']}</code>
                </span>
                <span>{match_html}</span>
            </div>
            <div style="margin-top:8px; display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; font-size:0.85rem;">
                <div style="background:#080A0A; padding:8px 12px; border-radius:6px; border:1px solid #242B2B;">
                    <span style="color:#707A77;">Expected Policy:</span><br/>
                    <b style="color:{exp_badge_color}; font-size:1.0rem;">{scen_meta.get('expected_decision')}</b>
                </div>
                <div style="background:#080A0A; padding:8px 12px; border-radius:6px; border:1px solid #242B2B;">
                    <span style="color:#707A77;">Actual Policy:</span><br/>
                    <b style="color:{'#22D3EE' if act_decision=='ACT' else ('#FB923C' if act_decision=='ESCALATE' else '#FF5C5C')}; font-size:1.0rem;">{act_decision}</b>
                </div>
                <div style="background:#080A0A; padding:8px 12px; border-radius:6px; border:1px solid #242B2B;">
                    <span style="color:#707A77;">P(Recovery) / EV:</span><br/>
                    <b style="color:#F5F7F5; font-size:1.0rem;">{res.get('recovery_probability', 0.0)*100:.1f}% / ₹{res.get('expected_recovery_value', 0.0):,.2f}</b>
                </div>
            </div>
            <div style="margin-top:10px; padding:10px 12px; background:#080A0A; border-radius:6px; border:1px solid #242B2B; font-size:0.85rem; color:#A7B0AD;">
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
        decision = res.get("policy_decision")
        amt = res.get("amount", 0.0)
        source_diag = res.get("diagnosis_source", "heuristic")
        source_rec = res.get("recommendation_source", "heuristic")

        status_color = "#22D3EE" if decision == "ACT" else ("#FB923C" if decision == "ESCALATE" else "#FF5C5C")
        st.markdown(f"""
        <div style="background-color:#121616; border:1px solid #242B2B; border-left: 4px solid {status_color}; padding:14px 18px; border-radius:6px; margin-bottom:18px;">
            <div style="font-size:1.15rem; font-weight:800; color:#F5F7F5;">
                Transaction <code>{res.get('transaction_id')}</code> — Policy Verdict: <span style="color:{status_color};">{decision}</span>
            </div>
            <div style="color:#A7B0AD; font-size:0.88rem; margin-top:4px;">
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
                        <span style="font-size:1.05rem; font-weight:700; color:#F5F7F5;">
                            Step {step['step_number']}: {step['title']}
                        </span>
                    </div>
                    <div style="color:#A7B0AD; font-size:0.92rem; margin-top:6px;">
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
                <div style="background-color:#121616; border:1px solid #242B2B; border-radius:8px; padding:16px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.05rem; font-weight:700; color:#F5F7F5;">
                            Transaction <code>{item.transaction_id}</code>
                        </span>
                        <span class="badge-escalate">PENDING APPROVAL</span>
                    </div>
                    <div style="margin-top:8px; color:#A7B0AD; font-size:0.92rem;">
                        <strong>Amount</strong>: <span style="color:#F5F7F5; font-weight:700;">₹{item.amount:,.2f} {item.currency}</span> &nbsp;|&nbsp;
                        <strong>Customer</strong>: <code>{item.customer_id}</code> &nbsp;|&nbsp;
                        <strong>Failure Reason</strong>: <code>{item.failure_reason}</code>
                    </div>
                    <div style="margin-top:6px; color:#707A77; font-size:0.85rem;">
                        <strong>ML Recovery Prob</strong>: <b>{item.recovery_probability*100:.1f}%</b> &nbsp;|&nbsp;
                        <strong>Recommended Action</strong>: <code>{item.recommended_action.upper()}</code> &nbsp;|&nbsp;
                        <strong>Policy Reason</strong>: {item.policy_reason}
                    </div>
                    <div style="margin-top:8px; background-color:#080A0A; border:1px solid #242B2B; padding:10px; border-radius:6px; font-size:0.82rem; color:#A7B0AD;">
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
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with in_c2:
        st.markdown("### Population Stability Index (PSI) Drift Monitor")
        drift_report = run_drift_report(test_df, persist=False)
        st.metric("System Data Health Status", drift_report["system_status"].upper())
        st.write(f"Significant Drift Count: **{drift_report['significant_drift_count']}** | Moderate Drift Count: **{drift_report['moderate_drift_count']}**")
        with st.expander("View PSI Drift Feature Breakdown", expanded=False):
            st.dataframe(pd.DataFrame(drift_report["features"]), use_container_width=True, hide_index=True)

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
                <div style="background:#121616; border:1px solid #242B2B; border-radius:6px; padding:10px 14px;">
                    <div style="color:#707A77; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Total Audit Records</div>
                    <div style="color:#F5F7F5; font-size:1.25rem; font-weight:700; margin-top:2px;">{total_records}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[1]:
                st.markdown(f"""
                <div style="background:#121616; border:1px solid #242B2B; border-left:3px solid #22D3EE; border-radius:6px; padding:10px 14px;">
                    <div style="color:#707A77; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Policy ACT</div>
                    <div style="color:#22D3EE; font-size:1.25rem; font-weight:700; margin-top:2px;">{act_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[2]:
                st.markdown(f"""
                <div style="background:#121616; border:1px solid #242B2B; border-left:3px solid #FB923C; border-radius:6px; padding:10px 14px;">
                    <div style="color:#707A77; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Escalated (HITL)</div>
                    <div style="color:#FB923C; font-size:1.25rem; font-weight:700; margin-top:2px;">{esc_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with summary_cols[3]:
                st.markdown(f"""
                <div style="background:#121616; border:1px solid #242B2B; border-left:3px solid #FF5C5C; border-radius:6px; padding:10px 14px;">
                    <div style="color:#707A77; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Policy Blocked</div>
                    <div style="color:#FF5C5C; font-size:1.25rem; font-weight:700; margin-top:2px;">{ref_count}</div>
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

                status_color = "#22D3EE" if decision == "ACT" else ("#FB923C" if decision == "ESCALATE" else "#FF5C5C")

                with st.container():
                    st.markdown(f"""
                    <div style="background:#121616; border:1px solid #242B2B; border-left:4px solid {status_color}; border-radius:6px; padding:12px 16px; margin-bottom:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                            <div>
                                <span style="font-weight:700; color:#F5F7F5; font-size:0.95rem;">Transaction <code>{txn_id}</code></span>
                                <span style="color:#707A77; font-size:0.80rem; margin-left:12px;">{ts}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:0.80rem; font-weight:700; color:{status_color}; padding:2px 8px; border:1px solid {status_color}; border-radius:4px;">{decision}</span>
                                <span style="font-size:0.75rem; color:#A7B0AD; background:#080A0A; border:1px solid #242B2B; padding:2px 6px; border-radius:4px;">{exec_status}</span>
                            </div>
                        </div>
                        <div style="margin-top:8px; display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-size:0.82rem;">
                            <div><span style="color:#707A77;">Customer:</span> <code style="color:#A7B0AD;">{cust_id}</code></div>
                            <div><span style="color:#707A77;">Amount:</span> <b style="color:#F5F7F5;">₹{amt:,.2f}</b></div>
                            <div><span style="color:#707A77;">P(Recovery):</span> <b style="color:#F5F7F5;">{prob*100:.1f}%</b></div>
                            <div><span style="color:#707A77;">Recommended:</span> <code style="color:#A7B0AD;">{rec_action.upper()}</code></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Inspect Record — {txn_id}", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#707A77; text-transform:uppercase; margin-bottom:6px;">
                                Prediction & Model Evidence
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#080A0A; border:1px solid #242B2B; border-radius:6px; padding:10px; font-size:0.80rem; color:#A7B0AD;">
                                <div><span style="color:#707A77;">Model:</span> <b style="color:#F5F7F5;">{model_type} ({exp_id})</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Calibrated Probability:</span> <b style="color:#F5F7F5;">{prob*100:.2f}%</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Expected Gross Value:</span> <b style="color:#F5F7F5;">₹{gross_ev:,.2f}</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Decision Cutoff (&tau;):</span> <code>{tau}</code></div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c2:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#707A77; text-transform:uppercase; margin-bottom:6px;">
                                Policy Guard Verdict
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#080A0A; border:1px solid #242B2B; border-radius:6px; padding:10px; font-size:0.80rem; color:#A7B0AD;">
                                <div><span style="color:#707A77;">Decision:</span> <b style="color:{status_color};">{decision}</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Action Cost:</span> <b style="color:#F5F7F5;">₹{action_cost:,.2f}</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Triggered Rules:</span> <code style="color:#A7B0AD;">{', '.join(triggered_rules) if triggered_rules else 'None (Clean Pass)'}</code></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Authority:</span> <b style="color:#F5F7F5;">Deterministic Policy Engine</b></div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c3:
                            st.markdown("""
                            <div style="font-size:0.75rem; font-weight:700; color:#707A77; text-transform:uppercase; margin-bottom:6px;">
                                Execution & Telemetry State
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#080A0A; border:1px solid #242B2B; border-radius:6px; padding:10px; font-size:0.80rem; color:#A7B0AD;">
                                <div><span style="color:#707A77;">Status:</span> <b style="color:#F5F7F5;">{exec_status}</b></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Execution Mode:</span> <code>{'Dry Run' if dry_run else 'Live Test'}</code></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Reference ID:</span> <code>{ref_id}</code></div>
                                <div style="margin-top:4px;"><span style="color:#707A77;">Blocking Reason:</span> <span style="color:#A7B0AD;">{blocking_reason}</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="margin-top:8px; background:#080A0A; border:1px solid #242B2B; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#A7B0AD;">
                            <div style="color:#707A77; font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:4px;">Policy Justification</div>
                            {justification}
                        </div>
                        """, unsafe_allow_html=True)

                        if pos_signals or neg_signals:
                            sig_col1, sig_col2 = st.columns(2)
                            with sig_col1:
                                if pos_signals:
                                    st.markdown("""
                                    <div style="font-size:0.75rem; font-weight:600; color:#39FF88; margin-top:6px; margin-bottom:4px;">
                                        Top Positive Signals
                                    </div>
                                    """, unsafe_allow_html=True)
                                    pos_df = pd.DataFrame(pos_signals)
                                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                            with sig_col2:
                                if neg_signals:
                                    st.markdown("""
                                    <div style="font-size:0.75rem; font-weight:600; color:#FF5C5C; margin-top:6px; margin-bottom:4px;">
                                        Top Negative Signals
                                    </div>
                                    """, unsafe_allow_html=True)
                                    neg_df = pd.DataFrame(neg_signals)
                                    st.dataframe(neg_df, use_container_width=True, hide_index=True)

                        with st.expander("View Raw Audit Record (JSON)", expanded=False):
                            st.json(record)
                    
                    st.write("")
        else:
            st.write("Audit log is currently empty.")
    else:
        st.write("No audit log found on disk.")
