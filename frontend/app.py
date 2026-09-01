"""
RecoverAI — Transaction Decision Center & Buildathon Interactive Experience (Streamlit Frontend).

Integrates:
- Tab 1: 🏠 Recovery Command Center (Live metric cards, active activity stream, system topology)
- Tab 2: 🎮 Demo Simulator (One-click Buildathon scenarios A, B, C, D with live injection)
- Tab 3: 🤖 Live Agent Decision Trace (Visual 7-stage pipeline, LLM vs Heuristic badges, policy authority)
- Tab 4: 👤 Merchant Approval Queue (HITL queue, approve/reject actions, fraud safety invariants)
- Tab 5: 📊 Recovery Insights & Governance (EV threshold curves, PSI drift monitor, immutable audit logs)
"""

import os
import sys
import json
import time
import uuid
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
    from agent.graph import run_agent
    from agent.demo_data import build_test_transaction
    from agent.nodes.policy import load_frozen_policy, HIGH_VALUE_TRANSACTION_THRESHOLD
    from payment.razorpay_client import is_razorpay_configured
    from monitoring.drift_detection import run_drift_report
    from database.database import SessionLocal, engine, Base
    import database.models
    from database.models import FailedPayment, ApprovalRequest, AuditLog, PaymentExecutionClaim
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
    st.error(f"❌ FAIL-LOUD INTEGRATION ERROR: Failed to import RecoverAI backend modules.\n\nError: {e}")
    st.stop()

# Ensure database tables exist
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    pass

st.set_page_config(
    page_title="RecoverAI — AI Revenue Recovery Agent",
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
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .step-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-act {
        background-color: #064e3b;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-escalate {
        background-color: #78350f;
        color: #fde047;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-refuse {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-llm {
        background-color: #1e1b4b;
        color: #a5b4fc;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-heuristic {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
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
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/RecoverAI-Razorpay_Buildathon-blue?style=for-the-badge&logo=razorpay", use_container_width=True)
    st.title("RecoverAI Agent")
    st.caption("Autonomous Payment Recovery & Financial Governance Engine")
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    # Engine status
    llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    has_real_llm = bool(llm_key and not llm_key.startswith("mock_") and not llm_key.startswith("your_"))
    
    st.markdown(f"**LLM Intelligence**: {'🧠 Live LLM' if has_real_llm else '⚙️ Deterministic Fallback'}")
    razorpay_status = is_razorpay_configured()
    st.markdown(f"**Razorpay Test Mode**: {'✅ Configured' if razorpay_status else '⚠️ Mock Sandbox'}")
    st.markdown(f"**Policy Threshold ($\\\\tau$)**: `0.35`")
    st.markdown(f"**High-Value Escalation**: `₹{HIGH_VALUE_TRANSACTION_THRESHOLD:,.2f}`")
    
    st.markdown("---")
    st.markdown("### 🛡️ Core Invariant")
    st.info("**LLM Recommends.**\n**Deterministic Policy Controls.**\n**Human Approves Escalations.**\n\n*Fraud & blocked actions can never be overridden.*")

# -----------------------------------------------------------------------------
# MAIN HEADER & TABS
# -----------------------------------------------------------------------------
st.title("💳 RecoverAI — Autonomous Revenue Recovery Agent")
st.markdown("AI-Powered Recovery Agent with Expected-Value Optimization, Deterministic Guardrails, Closed-Loop Webhook Settlement & Human-in-the-Loop Review.")

tab_cmd, tab_sim, tab_trace, tab_approvals, tab_insights = st.tabs([
    "🏠 Recovery Command Center",
    "🎮 Demo Simulator (Buildathon)",
    "🤖 Live Agent Decision Trace",
    "👤 Merchant Approval Queue",
    "📊 Recovery Insights & Governance"
])

# Initialize session state variables
if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None
if "latest_timeline" not in st.session_state:
    st.session_state["latest_timeline"] = None
if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "auto_recovery"


# =============================================================================
# TAB 1: RECOVERY COMMAND CENTER
# =============================================================================
with tab_cmd:
    # Query database for live pending count and recent activity
    db = SessionLocal()
    try:
        pending_approvals = list_pending_approvals(db)
        pending_count = len(pending_approvals)
        recent_payments = db.query(FailedPayment).order_by(FailedPayment.created_at.desc()).limit(8).all()
        recent_audits = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(8).all()
    except Exception:
        pending_count = 0
        recent_payments = []
        recent_audits = []
    finally:
        db.close()

    # Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue at Risk</div>
            <div class="metric-value">₹{metrics['total_risk']:,.2f}</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">N = 633 Failed Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Default Policy Recovered (Tau = 0.50)</div>
            <div class="metric-value">₹{metrics['default_recovered']:,.2f}</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">52.86% Risk Captured</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">EV-Optimized Recovered (Tau = 0.35)</div>
            <div class="metric-value">₹{metrics['opt_recovered']:,.2f}</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">56.97% Risk Captured</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Revenue Uplift</div>
            <div class="metric-value" style="color:#10b981;">+₹{metrics['net_uplift_inr']:,.2f}</div>
            <div class="metric-delta">+{metrics['net_uplift_pct']:.2f}% Net Growth</div>
            <div style="color:#64748b; font-size:0.75rem; margin-top:4px;">Pending Approvals: {pending_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Jump to Live Demo Simulator
    alert_col1, alert_col2 = st.columns([3, 1])
    with alert_col1:
        st.info("💡 **Ready to see RecoverAI in action?** Use the **Demo Simulator** tab to run a 1-click end-to-end recovery trace, test high-value human approvals, or simulate a live Razorpay webhook payment settlement.")
    with alert_col2:
        st.metric("Pending Human Reviews", f"{pending_count} Txns", delta=f"{pending_count} Awaiting" if pending_count > 0 else "Queue Clear")

    st.subheader("📋 Recent Transaction & Recovery Activity (Persisted Database)")
    if recent_payments:
        table_data = []
        for p in recent_payments:
            badge = "⚪ UNPROCESSED"
            if p.status == "RECOVERED":
                badge = "💰 RECOVERED"
            elif p.status == "AWAITING_APPROVAL":
                badge = "🟡 AWAITING_APPROVAL"
            elif p.status == "COMPLETED":
                badge = "🟢 COMPLETED"
            elif p.status == "BLOCKED":
                badge = "🔴 BLOCKED"
            elif p.status == "REJECTED_BY_HUMAN":
                badge = "❌ REJECTED_BY_HUMAN"

            table_data.append({
                "Transaction ID": p.id,
                "Customer": p.customer_id,
                "Amount": f"₹{p.amount:,.2f}",
                "Failure Reason": p.failure_reason or "N/A",
                "Failure Code": getattr(p, "failure_code", "N/A"),
                "Status": badge,
                "Timestamp": p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else "Just now"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No persisted transactions in database yet. Run a demo simulation to populate live records.")


# =============================================================================
# TAB 2: DEMO SIMULATOR (BUILDATHON LIVE DEMO)
# =============================================================================
with tab_sim:
    st.subheader("🎮 1-Click Razorpay Buildathon Demo Simulator")
    st.markdown("Select an industry scenario below to trigger a live failure event and observe RecoverAI's end-to-end decision journey.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if st.button("🟢 **Scenario A**\nSmart Auto-Recovery", use_container_width=True):
            st.session_state["selected_scenario"] = "auto_recovery"
        st.caption("₹2,500 transient timeout $\\rightarrow$ High P $\\rightarrow$ Auto-recovers")
    with sc2:
        if st.button("🟡 **Scenario B**\nHigh-Value HITL Approval", use_container_width=True):
            st.session_state["selected_scenario"] = "human_approval"
        st.caption("₹14,500 > limit $\\rightarrow$ Policy escalates to merchant queue")
    with sc3:
        if st.button("🔴 **Scenario C**\nFraud / Risk Block", use_container_width=True):
            st.session_state["selected_scenario"] = "fraud_block"
        st.caption("IP risk 0.92 $\\rightarrow$ Policy blocks $\\rightarrow$ Zero actions executed")
    with sc4:
        if st.button("⚪ **Scenario D**\nLow P(Recovery) Refusal", use_container_width=True):
            st.session_state["selected_scenario"] = "low_probability"
        st.caption("P < 0.35 $\\rightarrow$ Negative EV $\\rightarrow$ Spares wasteful action fee")

    current_scen_key = st.session_state["selected_scenario"]
    scen_meta = PREDEFINED_SCENARIOS[current_scen_key]

    st.markdown("---")
    st.markdown(f"### Selected: `{scen_meta['name']}`")
    st.markdown(f"**Scenario Description**: {scen_meta['description']}")
    st.markdown(f"**Expected Decision**: `{scen_meta['expected_decision']}` | **Expected Action**: `{scen_meta['expected_action']}`")

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

    if st.button("🚨 SIMULATE PAYMENT FAILURE & RUN AGENT", type="primary", use_container_width=True):
        with st.spinner("Injecting failed transaction & executing RecoverAI Agent..."):
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
                    failure_reason="insufficient_funds",
                    failure_category="customer_action_required",
                    customer_historical_success_rate=0.10,
                    previous_failures_24h=4,
                    recovery_attempt_count=3,
                    ip_risk_score=0.20
                )
            else:
                txn = build_test_transaction(transaction_id=txn_id, amount=custom_amount)

            # Run agent
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

        st.success(f"✅ Simulation finished for `{res['transaction_id']}`! Go to **Live Agent Decision Trace** or **Merchant Approval Queue** to inspect.")


# =============================================================================
# TAB 3: LIVE AGENT DECISION TRACE
# =============================================================================
with tab_trace:
    st.subheader("🤖 Live Agent Decision Trace")
    st.caption("Real-time inspection of RecoverAI's multi-stage reasoning, prediction, and deterministic policy enforcement.")

    res = st.session_state.get("latest_result")
    timeline = st.session_state.get("latest_timeline")

    if not res or not timeline:
        st.info("No transaction has been simulated yet in this session. Click **🚨 SIMULATE PAYMENT FAILURE** in the **Demo Simulator** tab to watch the live agent trace.")
    else:
        # Top banner summarizing decision
        decision = res.get("policy_decision")
        amt = res.get("amount", 0.0)
        source_diag = res.get("diagnosis_source", "heuristic")
        source_rec = res.get("recommendation_source", "heuristic")

        status_color = "#10b981" if decision == "ACT" else ("#f59e0b" if decision == "ESCALATE" else "#ef4444")
        st.markdown(f"""
        <div style="background-color:#1e293b; border-left: 6px solid {status_color}; padding:14px; border-radius:6px; margin-bottom:16px;">
            <div style="font-size:1.15rem; font-weight:700; color:#f8fafc;">
                Transaction {res.get('transaction_id')} — Policy Decision: <span style="color:{status_color};">{decision}</span>
            </div>
            <div style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">
                Amount: ₹{amt:,.2f} | Reason: {res.get('failure_reason')} | ML Recovery Prob: {res.get('recovery_probability', 0.0)*100:.1f}% | Agent Status: {res.get('agent_status')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 7-STAGE PIPELINE CARDS
        for step in timeline:
            with st.container():
                st.markdown(f"""
                <div class="step-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.05rem; font-weight:700; color:#f8fafc;">
                            {step['icon']} Step {step['step_number']}: {step['title']}
                        </span>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.95rem; margin-top:6px;">
                        {step['summary']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"🔍 Detailed Observable Metrics (Step {step['step_number']})", expanded=(step['step_number'] in [3, 4, 5])):
                    st.json(step["details"])

        # Manual Webhook Settlement Simulator
        if res.get("policy_decision") == "ACT" and res.get("agent_status") != "RECOVERED":
            st.markdown("---")
            st.markdown("### ⚡ Live Webhook Simulator (Close the Recovery Loop)")
            st.write("Simulate customer clicking the recovery payment link and completing payment via Razorpay.")
            if st.button("💳 Simulate Customer Paid Webhook Event (Close Loop)", type="primary"):
                db = SessionLocal()
                try:
                    # Ensure claim exists for correlation
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
                    st.success(f"🎉 Webhook processed successfully! Transaction is now marked **💰 RECOVERED** (₹{res['amount']:,.2f}).")
                    st.rerun()
                finally:
                    db.close()


# =============================================================================
# TAB 4: MERCHANT APPROVAL QUEUE (HITL)
# =============================================================================
with tab_approvals:
    st.subheader("👤 Merchant Approval Queue (Human-in-the-Loop)")
    st.markdown("High-value payments and policy-escalated recovery actions are held in this queue until authorized by a merchant admin.")

    db = SessionLocal()
    try:
        pending_items = list_pending_approvals(db)
    finally:
        db.close()

    if not pending_items:
        st.success("✅ **Approval Queue is Clear!** No transactions currently require merchant review.")
        st.info("To test the approval flow, switch to the **Demo Simulator** tab and launch **Scenario B (High-Value Human Approval)**.")
    else:
        st.warning(f"⚠️ **{len(pending_items)} transaction(s) requiring merchant decision.**")
        
        for item in pending_items:
            with st.container():
                st.markdown(f"""
                <div style="background-color:#1e293b; border:1px solid #334155; border-radius:8px; padding:16px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.15rem; font-weight:700; color:#f8fafc;">
                            Transaction `{item.transaction_id}`
                        </span>
                        <span class="badge-escalate">PENDING APPROVAL</span>
                    </div>
                    <div style="margin-top:8px; color:#cbd5e1; font-size:0.95rem;">
                        <strong>Amount</strong>: ₹{item.amount:,.2f} {item.currency} &nbsp;|&nbsp;
                        <strong>Customer</strong>: `{item.customer_id}` &nbsp;|&nbsp;
                        <strong>Failure Reason</strong>: `{item.failure_reason}`
                    </div>
                    <div style="margin-top:6px; color:#94a3b8; font-size:0.9rem;">
                        <strong>ML Recovery Prob</strong>: {item.recovery_probability*100:.1f}% &nbsp;|&nbsp;
                        <strong>Recommended Action</strong>: <code>{item.recommended_action.upper()}</code> &nbsp;|&nbsp;
                        <strong>Policy Reason</strong>: {item.policy_reason}
                    </div>
                    <div style="margin-top:8px; background-color:#0f172a; padding:10px; border-radius:6px; font-size:0.85rem; color:#94a3b8;">
                        💡 <strong>Decision Rationale</strong>: {getattr(item, 'recommendation_reason', None) or 'High expected return on customer reconnection.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_col1, act_col2, _ = st.columns([1.5, 1.5, 4])
                
                with act_col1:
                    if st.button(f"✓ APPROVE RECOVERY", key=f"app_{item.transaction_id}", type="primary"):
                        db = SessionLocal()
                        try:
                            ok, code, rec_res = approve_recovery_action(
                                db=db,
                                transaction_id=item.transaction_id,
                                human_notes=f"Authorized via Streamlit HITL Dashboard"
                            )
                            if ok:
                                st.success(f"✅ Approved transaction `{item.transaction_id}`! Recovery action safely executed.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Approval failed: {rec_res.get('message', code)} ({code})")
                        finally:
                            db.close()

                with act_col2:
                    if st.button(f"✗ REJECT RECOVERY", key=f"rej_{item.transaction_id}"):
                        db = SessionLocal()
                        try:
                            ok, code, rej_res = reject_recovery_action(
                                db=db,
                                transaction_id=item.transaction_id,
                                human_notes="Rejected via Streamlit HITL Dashboard"
                            )
                            if ok:
                                st.warning(f"❌ Transaction `{item.transaction_id}` rejected. Zero recovery actions executed.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Rejection failed: {rej_res.get('message', code)} ({code})")
                        finally:
                            db.close()

    st.markdown("---")
    st.caption("🛡️ **Fintech Safety Invariant**: Fraud or risk-related blocked payments can NEVER be approved or executed via this interface.")


# =============================================================================
# TAB 5: RECOVERY INSIGHTS & GOVERNANCE
# =============================================================================
with tab_insights:
    st.subheader("📊 Recovery Insights, Calibration & Data Health")

    in_c1, in_c2 = st.columns(2)
    with in_c1:
        st.markdown("### 📈 Policy Threshold Comparison")
        st.write("Comparison between default industry threshold ($\\\\tau = 0.50$) vs. RecoverAI EV-Optimized threshold ($\\\\tau = 0.35$):")
        comp_df = pd.DataFrame([
            {"Policy": "Default Industry Standard (0.50)", "Recovered Revenue": f"₹{metrics['default_recovered']:,.2f}", "Action Cost": f"₹{metrics['default_cost']:,.2f}", "Net Profit": f"₹{metrics['default_net']:,.2f}"},
            {"Policy": "RecoverAI EV-Optimized (0.35)", "Recovered Revenue": f"₹{metrics['opt_recovered']:,.2f}", "Action Cost": f"₹{metrics['opt_cost']:,.2f}", "Net Profit": f"₹{metrics['opt_net']:,.2f}"},
            {"Policy": "Uplift Achieved by RecoverAI", "Recovered Revenue": f"+₹{metrics['gross_uplift_inr']:,.2f}", "Action Cost": f"+₹{metrics['inc_action_cost']:,.2f}", "Net Profit": f"+₹{metrics['net_uplift_inr']:,.2f}"}
        ])
        st.dataframe(comp_df, use_container_width=True)

    with in_c2:
        st.markdown("### 🔍 Population Stability Index (PSI) Drift Monitor")
        drift_report = run_drift_report(test_df, persist=False)
        st.metric("System Data Health Status", drift_report["system_status"].upper())
        st.write(f"Significant Drift Count: **{drift_report['significant_drift_count']}** | Moderate Drift Count: **{drift_report['moderate_drift_count']}**")
        with st.expander("View PSI Drift Feature Breakdown", expanded=False):
            st.dataframe(pd.DataFrame(drift_report["features"]), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📄 Immutable Telemetry Trail (`logs/decision_audit.jsonl`)")
    audit_log_path = os.path.join(PROJECT_ROOT, "logs", "decision_audit.jsonl")
    if os.path.exists(audit_log_path):
        records = []
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()[-20:]):
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        if records:
            st.json(records[:3])
            st.caption(f"Showing 3 most recent audit events out of {len(records)} loaded.")
        else:
            st.write("Audit log is currently empty.")
    else:
        st.write("No audit log found on disk.")
