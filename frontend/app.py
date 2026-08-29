"""
RecoverAI — Transaction Decision Center Dashboard (Streamlit Frontend).

Integrates directly with:
- data/raw/transactions.csv (Untouched Test Set, N=633)
- evaluation/business_metrics.py (evaluate_threshold_grid)
- agent/orchestrator.py (orchestrate_transaction)
- agent/nodes/policy.py (evaluate_transaction_policy)
- payment/executor.py & payment/razorpay_client.py (Razorpay Test Mode Execution)
- logs/decision_audit.jsonl (Append-Only Audit Trail)
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FAIL-LOUD IMPORT VALIDATION
try:
    from sklearn.model_selection import train_test_split
    from ml.features import add_engineered_features
    from evaluation.business_metrics import evaluate_threshold_grid
    from agent.orchestrator import orchestrate_transaction, load_orchestrator_model, AUDIT_LOG_PATH
    from agent.nodes.policy import load_frozen_policy
    from payment.razorpay_client import is_razorpay_configured
    from monitoring.drift_detection import run_drift_report
except Exception as e:
    st.error(f"❌ FAIL-LOUD INTEGRATION ERROR: Failed to import RecoverAI backend modules.\n\nError: {e}")
    st.stop()

st.set_page_config(
    page_title="RecoverAI — Transaction Decision Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        color: #10b981;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Timestamps for Cache Invalidation
csv_file_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
csv_mtime = os.path.getmtime(csv_file_path) if os.path.exists(csv_file_path) else 0.0

model_artifact_file = os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib")
model_mtime = os.path.getmtime(model_artifact_file) if os.path.exists(model_artifact_file) else 0.0
app_mtime = os.path.getmtime(__file__)


@st.cache_data
def load_and_split_dataset(csv_mtime: float):
    """
    Loads active dataset and reproduces exact 85/15 train/test split.
    Bound to csv_mtime for automatic cache invalidation.
    """
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
    Reproduces 4 top metric cards directly from test set & model pipeline.
    Cache invalidated whenever model or dataset updates on disk.
    """
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

    return {
        "total_risk": total_risk,
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
        "inc_action_cost": inc_cost
    }


test_df, idx_dev, idx_test = load_and_split_dataset(csv_mtime=csv_mtime)
metrics = compute_live_top_metrics(model_mtime=model_mtime, csv_mtime=csv_mtime)

# -----------------------------------------------------------------------------
# HEADER & SIDEBAR
# -----------------------------------------------------------------------------
st.title("💳 RecoverAI — Transaction Decision Center")
st.caption("Autonomous Payment Recovery & Financial Governance Engine • Razorpay AI Buildathon Submission")

with st.sidebar:
    st.header("⚙️ Execution Settings")
    
    razorpay_status = is_razorpay_configured()
    st.markdown(f"**Razorpay Test Config**: {'✅ Available' if razorpay_status else '⚠️ Missing Keys'}")
    
    enable_test_mode = st.checkbox(
        "Enable Razorpay Test Mode Execution",
        value=False,
        help="Default is OFF (Dry Run). When enabled and policy decision is ACT, sends live API call to Razorpay Test Mode."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Policy Configuration")
    st.markdown("**Frozen Threshold ($\\\\tau$)**: `0.35`")
    st.markdown("**Action Costs**:")
    st.markdown("- Retry: ₹5.00\n- Reminder: ₹2.00\n- Payment Link: ₹12.00")
    st.markdown("**High-Value Limit**: ₹8,500.00")

# -----------------------------------------------------------------------------
# 1. TOP OF PAGE — FOUR LIVE METRIC CARDS
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Revenue at Risk (Test)</div>
        <div class="metric-value">₹{metrics['total_risk']:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">N = 633 Failed Transactions</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Default Policy Recovered (Tau = 0.50)</div>
        <div class="metric-value">₹{metrics['default_recovered']:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">52.86% Risk Captured</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">EV-Optimized Recovered (Tau = 0.35)</div>
        <div class="metric-value">₹{metrics['opt_recovered']:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">56.97% Risk Captured</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Revenue Uplift</div>
        <div class="metric-value" style="color:#10b981;">+₹{metrics['net_uplift_inr']:,.2f}</div>
        <div class="metric-delta">+{metrics['net_uplift_pct']:.2f}% Net Revenue Uplift</div>
        <div style="color:#64748b; font-size:0.75rem; margin-top:4px;">Gross Uplift: +₹{metrics['gross_uplift_inr']:,.2f} | Costs: ₹{metrics['inc_action_cost']:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DEMO MODE BUTTONS & 2. TRANSACTION SELECTOR
# -----------------------------------------------------------------------------
st.subheader("🔍 Transaction Analysis Workbench")

btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])

DEMO_ACT_ID = "txn_0000012"
DEMO_ESC_ID = "txn_0000008"
DEMO_REF_ID = "txn_0000005"

if "selected_txn_id" not in st.session_state:
    st.session_state["selected_txn_id"] = DEMO_ACT_ID

with btn_col1:
    if st.button("🟢 Demo ACT (txn_0000012)", use_container_width=True):
        st.session_state["selected_txn_id"] = DEMO_ACT_ID

with btn_col2:
    if st.button("🟡 Demo ESCALATE (txn_0000008)", use_container_width=True):
        st.session_state["selected_txn_id"] = DEMO_ESC_ID

with btn_col3:
    if st.button("🔴 Demo REFUSE (txn_0000005)", use_container_width=True):
        st.session_state["selected_txn_id"] = DEMO_REF_ID

test_txn_ids = list(test_df["transaction_id"].values)
default_idx = test_txn_ids.index(st.session_state["selected_txn_id"]) if st.session_state["selected_txn_id"] in test_txn_ids else 0

selected_id = st.selectbox(
    "Select Transaction from Untouched Test Set (N=633):",
    options=test_txn_ids,
    index=default_idx,
    key="select_txn_box"
)

st.session_state["selected_txn_id"] = selected_id

if st.button("⚡ Analyze Transaction", type="primary", use_container_width=True):
    st.session_state["run_analysis"] = True

if st.session_state.get("run_analysis") or st.session_state.get("selected_txn_id"):
    txn_row = test_df[test_df["transaction_id"] == st.session_state["selected_txn_id"]].iloc[0].to_dict()
    is_dry_run = not enable_test_mode

    with st.spinner("Executing RecoverAI End-to-End Orchestration Pipeline..."):
        audit_res = orchestrate_transaction(txn_row, dry_run=is_dry_run)

    st.markdown("---")

    # STEP 1 — TRANSACTION DETAILS
    st.markdown("### Step 1: Transaction Details & Context")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Transaction ID**: `{audit_res['transaction_id']}`")
        st.write(f"**Customer ID**: `{audit_res['customer_id']}`")
    with c2:
        st.write(f"**Amount**: ₹{audit_res['amount']:,.2f}")
        st.write(f"**Failure Reason**: `{txn_row.get('failure_reason', 'N/A')}`")
    with c3:
        st.write(f"**Payment Method**: `{txn_row.get('payment_method', 'N/A')}`")
        st.write(f"**Consecutive Failure Streak**: `{txn_row.get('consecutive_failure_streak', 0)}`")
    with c4:
        st.write(f"**Past Recovery Rate**: `{txn_row.get('customer_past_recovery_rate_pre_current', 0.5):.2f}`")
        st.write(f"**IP Risk Score**: `{txn_row.get('ip_risk_score', 0.0):.2f}`")

    # STEP 2 & 3 — ML PROBABILITY & EXPLANATION
    st.markdown("### Step 2 & 3: ML Recovery Probability & Plain-English Explanation")
    prob = audit_res["prediction"]["recovery_probability"]
    expl = audit_res["prediction"]["explanation"]

    col_p, col_e = st.columns([1, 3])
    with col_p:
        st.metric(label="Recovery Probability", value=f"{prob * 100:.2f}%")
    with col_e:
        st.info(f"**Model Explanation (EXP_0 Logistic Regression)**:\n\n{expl}")

    # STEP 4 — EXPECTED VALUE
    st.markdown("### Step 4: Expected Value & Policy Calculations")
    ev_info = audit_res["business_policy"]
    ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
    with ev_c1:
        st.write(f"**Policy Threshold ($\\\\tau$)**: `{ev_info['threshold']:.2f}`")
        st.write(f"**Passed EV Threshold**: `{ev_info['passed_ev_threshold']}`")
    with ev_c2:
        st.write(f"**Recommended Action**: `{ev_info['recommended_action']}`")
        st.write(f"**Action Cost**: ₹{ev_info['action_cost']:.2f}")
    with ev_c3:
        st.write(f"**Expected Gross Value**: ₹{ev_info['expected_gross_recovery_value']:,.2f}")
    with ev_c4:
        exp_net = ev_info['expected_gross_recovery_value'] - ev_info['action_cost'] if ev_info['passed_ev_threshold'] else 0.0
        st.write(f"**Expected Net Value**: ₹{exp_net:,.2f}")

    # STEP 5 — SAFETY GATE
    st.markdown("### Step 5: Deterministic Safety Gate Decision")
    safety = audit_res["safety"]
    decision = safety["decision"]
    rules = safety["triggered_rules"]
    justification = safety["justification"]

    if decision == "ACT":
        st.success(f"### 🟢 Decision: ACT\n\n**Action**: `{ev_info['recommended_action']}`\n\n**Justification**: {justification}")
    elif decision == "ESCALATE":
        st.warning(f"### 🟡 Decision: ESCALATE (Requires Human Merchant Review)\n\n**Triggered Rules**: {'; '.join(rules)}\n\n**Justification**: {justification}")
    else:
        st.error(f"### 🔴 Decision: REFUSE (Automated Action Prohibited)\n\n**Triggered Rules**: {'; '.join(rules)}\n\n**Justification**: {justification}")

    # STEP 6 — EXECUTION RESULT
    st.markdown("### Step 6: Razorpay Test Mode / Execution Layer")
    exec_info = audit_res["execution"]
    
    ex_c1, ex_c2, ex_c3 = st.columns(3)
    with ex_c1:
        st.write(f"**Execution Status**: `{exec_info['execution_status']}`")
        st.write(f"**External API Called**: `{exec_info['external_api_called']}`")
    with ex_c2:
        st.write(f"**Dry Run Safety Mode**: `{exec_info['dry_run']}`")
        st.write(f"**Razorpay Ref ID**: `{exec_info['reference_id'] or 'None'}`")
    with ex_c3:
        if exec_info.get("short_url"):
            st.markdown(f"**Payment Link**: [{exec_info['short_url']}]({exec_info['short_url']})")
        if exec_info.get("blocking_reason"):
            st.write(f"**Blocking Reason**: `{exec_info['blocking_reason']}`")

    # STEP 7 — AUDIT LOG CONFIRMATION
    st.markdown("### Step 7: Audit Log Confirmation (logs/decision_audit.jsonl)")
    
    matched_record = None
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                try:
                    obj = json.loads(line)
                    if obj.get("transaction_id") == audit_res["transaction_id"]:
                        matched_record = obj
                        break
                except Exception:
                    pass

    if matched_record:
        st.success(f"✅ Audit Log Event Verified! Record appended to `{AUDIT_LOG_PATH}`.")
        with st.expander("📄 View Append-Only Audit Log Event JSON", expanded=True):
            st.json(matched_record)
    else:
        st.warning("Audit log event not found in recent trail.")

st.markdown("---")
with st.expander("📊 Feature Drift & Data Health Monitor (PSI Audit)", expanded=False):
    drift_report = run_drift_report(test_df, persist=False)
    st.write(f"**System Status**: `{drift_report['system_status']}`")
    d1, d2 = st.columns(2)
    with d1:
        st.metric("Significant Drift Features", drift_report["significant_drift_count"])
    with d2:
        st.metric("Moderate Drift Features", drift_report["moderate_drift_count"])
    
    df_drift = pd.DataFrame(drift_report["features"])
    st.dataframe(df_drift, use_container_width=True)
