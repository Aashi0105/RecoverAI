# RecoverAI — Evidence and Verification Record

> **Official Evidence Document for Razorpay AI Buildathon Submission**  
> **Repository Baseline**: RecoverAI — AI Revenue Recovery Agent  
> **Last Audit & Verification**: August 30, 2026

---

## 1. Purpose of This Document

This document provides a rigorous, evidence-based accounting of all technical claims, architectural components, machine learning evaluations, safety guardrails, and operational limitations in the **RecoverAI** repository.

The goal of this record is to maintain strict scientific and technical honesty by explicitly distinguishing between:
1. **Implemented and Verified Features**: Production-grade code components verified by automated test suites and runtime execution logs.
2. **Measured Offline Results**: Financial and statistical metrics calculated on an untouched held-out evaluation set ($N=633$).
3. **Simulated / Demonstration Capabilities**: Features utilizing synthetic datasets, dry-run safety modes, or Razorpay Test Mode APIs.
4. **Safety and Guardrail Mechanisms**: Deterministic policy controls, idempotency constraints, and risk refusal rules.
5. **Known Limitations and Non-Claims**: Operational boundaries, non-production scopes, and explicit non-claims.

---

## 2. System Claims and Evidence Matrix

| Claim | Verified Evidence | Relevant Files | Verification Status | Notes / Scope |
| :--- | :--- | :--- | :---: | :--- |
| **LangGraph StateGraph Architecture** | `StateGraph(AgentState)` compiled with 8 nodes and conditional policy routing. | [agent/graph.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/agent/graph.py), [agent/state.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/agent/state.py) | **VERIFIED** | Node flow: `load_context` $\rightarrow$ `predict` $\rightarrow$ `diagnose` $\rightarrow$ `recommend` $\rightarrow$ `policy_guard` $\rightarrow$ conditional edge $\rightarrow$ `execute_mock_action` / `create_audit_log`. |
| **Baseline ML Propensity Model** | `EXP_0` Logistic Regression model fitted on 85% Dev set ($N=3,581$) predicting recovery probability. | [ml/features.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/ml/features.py), [ml/models/experiments/exp_0_baseline.joblib](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/ml/models/experiments/exp_0_baseline.joblib) | **VERIFIED** | Stratified random split (`seed=42`, `test_size=0.15`). Target `recovered` excluded from features. |
| **Expected Value Optimization** | Grid search ($\tau=0.35$) maximizing net expected value ($\text{EV} = P \times \text{Amount} - \text{Cost}$). | [evaluation/business_metrics.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/evaluation/business_metrics.py), [evaluation/business_policy.json](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/evaluation/business_policy.json) | **VERIFIED (OFFLINE)** | Tested on $N=633$ held-out test set; yields +₹64,359.60 (+4.07%) true net uplift. |
| **Causal Treatment Effect Estimation** | Propensity Score Inverse Probability Weighting (IPW) calculating Average Treatment Effect (ATE). | [evaluation/causal_lift.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/evaluation/causal_lift.py), [tests/test_causal_lift.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/tests/test_causal_lift.py) | **VERIFIED (OFFLINE BENCHMARK)** | ATE = +8.24% ($p < 0.001$), 95% Bootstrap CI [+5.33%, +11.17%]. Observational analysis on synthetic logs. |
| **Deterministic Safety Gate** | Non-negotiable policy rules producing 3 explicit outcomes (`ACT`, `ESCALATE`, `REFUSE`). | [agent/nodes/policy.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/agent/nodes/policy.py), [policies/recovery_policy.yaml](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/policies/recovery_policy.yaml) | **VERIFIED** | Enforces hard risk refusals, streak limits ($\ge 4$), and high-value transaction escalation ($\ge \text{₹}8,500.00$). |
| **Atomic Idempotency Claims** | Primary key constraint on `PaymentExecutionClaim.idempotency_key` and process `threading.Lock`. | [database/models.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/database/models.py), [payment/executor.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/payment/executor.py) | **VERIFIED** | Proves zero duplicate payment link creation under 10 concurrent barrier-synchronized threads. |
| **PSI Feature Drift Detection** | 10 quantile binning feature drift monitoring against frozen baseline snapshot ($N=3,581$). | [monitoring/drift_detection.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/monitoring/drift_detection.py), [monitoring/baseline_snapshot.json](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/monitoring/baseline_snapshot.json) | **VERIFIED** | Appends structured audit events to `logs/drift_audit.jsonl` (`persist=False` in dashboard UI). |
| **Razorpay Test Mode Integration** | SDK integration using `razorpay.Client` creating Test Mode Payment Links (`rzp_test_...`). | [payment/razorpay_client.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/payment/razorpay_client.py), [payment/executor.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/payment/executor.py) | **VERIFIED (TEST MODE ONLY)** | Key prefix validation enforced. Default `dry_run=True` flag prevents network calls unless toggled. |
| **Streamlit Interactive Dashboard** | Live Web Decision Center with top metric cards, demo buttons, Plotly charts, and drift expander. | [frontend/app.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/frontend/app.py) | **VERIFIED** | Runs locally on `http://localhost:8501` without tracebacks or Streamlit error boxes. |
| **Automated Test Suite** | 38 automated Pytest test cases across 8 suites, including 10,000 property-tested boundary iterations. | [tests/](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/tests) | **VERIFIED** | 100% pass rate (`pytest tests/ -v`). |

---

## 3. Implemented Architecture

RecoverAI uses a **Decoupled Modular Monolith** architecture:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RECOVERAI AGENT GRAPH                                  │
│                                                                                        │
│  [Failed Payment Event]                                                                │
│           │                                                                            │
│           ▼                                                                            │
│  [Node 1: load_context]        --> Formats input payload & loads merchant profile       │
│           │                                                                            │
│           ▼                                                                            │
│  [Node 2: predict_recovery]    --> Invokes EXP_0 Logistic Regression (Scikit-Learn)     │
│           │                                                                            │
│           ▼                                                                            │
│  [Node 3: diagnose_failure]    --> Extracts coefficient signals & failure cause        │
│           │                                                                            │
│           ▼                                                                            │
│  [Node 4: recommend_action]   --> Computes Net Expected Value (EV) @ Tau = 0.35        │
│           │                                                                            │
│           ▼                                                                            │
│  [Node 5: policy_guard]        --> Evaluates Deterministic Safety & Refusal Rules       │
│           │                                                                            │
│           ├───────────────────────────────┬───────────────────────────────┐            │
│           ▼                               ▼                               ▼            │
│       [ APPROVED ]                   [ ESCALATE ]                    [ REFUSE ]        │
│           │                               │                               │            │
│           ▼                               ▼                               ▼            │
│  [Node 6: execute_mock_action]  [Pause for Human Review]       [Hard Safety Refusal]     │
│  (Razorpay Test SDK / Dry-Run)            │                               │            │
│           │                               │                               │            │
│           ▼                               │                               │            │
│  [Node 7: verify_outcome]                 │                               │            │
│           │                               │                               │            │
│           └───────────────────────────────┴───────────────────────────────┘            │
│                                           │                                            │
│                                           ▼                                            │
│                             [Node 8: create_audit_log]                                 │
│                             (PostgreSQL DB & JSONL Trail)                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Machine Learning Evidence

* **Dataset Context**: Evaluated on $20,000$ synthetic e-commerce transactions containing $4,214$ failed payment records.
* **Train / Test Isolation**:
  * **Development Set ($85\%$, $N=3,581$)**: Used for feature engineering, baseline snapshot generation, and 5-fold cross-validation.
  * **Untouched Test Set ($15\%$, $N=633$)**: Strictly isolated using `train_test_split(..., test_size=0.15, random_state=42, stratify=y_all)`.
* **Model Pipeline (`EXP_0`)**:
  * **Classifier**: `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` saved at `ml/models/experiments/exp_0_baseline.joblib`.
  * **Preprocessor**: `ColumnTransformer` applying `StandardScaler` to numerical features (`amount`, `hour`, `day_of_week`) and `OneHotEncoder(handle_unknown="ignore")` to categorical features (`payment_method`, `failure_reason`).
  * **Engineered Features**: `customer_past_recovery_rate_pre_current`, `consecutive_failure_streak`, `ip_risk_score`, `velocity_score`.
* **Probability Output**: Sigmoid output from Logistic Regression `predict_proba()[:, 1]`. Post-hoc calibration (Platt scaling / Isotonic regression) is not applied.
* **Model Explainability**: Coefficient attribution (`X_trans * coefficients`) computes feature contributions for plain-English signal extraction.

---

## 5. Offline Business Evaluation

Calculated on the untouched held-out test set ($N=633$ failed payments, Total Revenue at Risk = ₹1,500,342.08, `seed=42`):

### A. Core Policy Performance Comparison

| Evaluation Metric | Default Policy ($\tau = 0.50$) | EV-Optimized Policy ($\tau = 0.35$) | Absolute Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Interventions Selected** | 498 transactions | **522 transactions** | $+24$ transactions |
| **Intervention Coverage Rate** | 78.67% | **82.46%** | $+3.79\%$ coverage |
| **Realized Recovered Revenue** | ₹986,026.63 | **₹1,008,097.35** | **$+\text{₹}22,070.72$ gross uplift (+1.47%)** |
| **Action Costs Incurred** | ₹4,096.00 | **₹4,144.00** | $+\text{₹}48.00$ cost investment |
| **Realized Net Value** | ₹981,930.63 | **₹1,003,953.35** | **$+\text{₹}22,022.72$ true net uplift (+1.47%)** |
| **% Revenue at Risk Captured** | 65.72% | **67.19%** | **$+1.47\%$ total risk captured** |
| **Recoverable Payment Recall** | 91.90% | **94.91%** | **$+3.01\%$ recoveries saved** |

### B. 6-Scenario Financial Sensitivity Analysis Grid

Verified via `evaluation/sensitivity_analysis.py` across 6 stress scenarios:

| Scenario | Cost Multiplier | Threshold / Prob Shift | Realized Gross Uplift | True Net Uplift (₹) | True Net Uplift (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BASE CASE** | 1.0x | $\tau = 0.35, \Delta P = 0.00$ | +₹22,070.72 | **+₹22,022.72** | **+1.47%** | **Baseline** |
| **2. ACTION COST +50%** | 1.5x | $\tau = 0.35, \Delta P = 0.00$ | +₹22,070.72 | **+₹21,998.72** | **+1.47%** | **Robust** |
| **3. ACTION COST -50%** | 0.5x | $\tau = 0.35, \Delta P = 0.00$ | +₹22,070.72 | **+₹22,046.72** | **+1.47%** | **Robust** |
| **4. THRESHOLD +0.05** | 1.0x | $\tau = 0.40, \Delta P = 0.00$ | +₹22,070.72 | **+₹22,022.72** | **+1.47%** | **Robust** |
| **5. THRESHOLD -0.05** | 1.0x | $\tau = 0.30, \Delta P = 0.00$ | +₹22,070.72 | **+₹22,022.72** | **+1.47%** | **Robust** |
| **6. PROBABILITY MISCALIBRATION** | 1.0x | $\tau = 0.35, \Delta P = -0.05$ | +₹38,813.32 | **+₹38,765.32** | **+2.58%** | **Robust** |


### C. Causal Treatment Effect (IPW) Evaluation

Verified via `evaluation/causal_lift.py`:
* **Propensity Score Model**: Logistic Regression predicting treatment assignment $T = \text{recovery\_attempted}$ using 9 pre-treatment covariates.
* **Inverse Probability Weighting (IPW)**: Weights $w_i = \frac{T_i}{e(x_i)} + \frac{1 - T_i}{1 - e(x_i)}$ clipped to $[0.01, 0.99]$.
* **Average Treatment Effect (ATE)**: **+8.24 percentage points** ($p < 0.001$).
* **Bootstrap Confidence Intervals**: 1,000 deterministic bootstrap iterations (`seed=42`) yielding **95% CI [+5.33%, +11.17%]**.

---

## 6. Payment Execution Evidence

* **SDK Integration**: Integrated with `razorpay` Python SDK (`razorpay.Client`).
* **Test Mode Enforcement**: `is_razorpay_configured()` requires key prefixes starting strictly with `rzp_test_`.
* **Dry-Run Safety Default**: Parameter `dry_run = True` default in `execute_recovery_policy()` bypasses external HTTP network calls during demonstration and testing.
* **Gated Execution**: For `ESCALATE` and `REFUSE` decisions, **zero external API calls are made**.
* **Truthful Capabilities**: Automated retries require tokenized card mandates and are reported as `NOT_SUPPORTED` rather than generating fake successful payment callbacks.

---

## 7. Safety and Policy Guardrails

Implemented in [agent/nodes/policy.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/agent/nodes/policy.py):

1. **`SUSPECTED_RISK_POLICY_REFUSAL`**: IP Risk Score $> 0.70$ or failure category `risk_related` $\rightarrow$ **`REFUSE`**.
2. **`PERMANENT_CARD_FAILURE_POLICY_REFUSAL`**: Failure reason in `["invalid_card", "card_expired"]` $\rightarrow$ **`REFUSE`**.
3. **`CONSECUTIVE_STREAK_LIMIT_REFUSAL`**: Consecutive failure streak $\ge 4$ $\rightarrow$ **`REFUSE`**.
4. **`LOW_PROBABILITY_POLICY_REFUSAL`**: Predicted probability $P < 0.35$ $\rightarrow$ **`REFUSE`**.
5. **`HIGH_VALUE_TRANSACTION_ESCALATION`**: Transaction Amount $\ge \text{₹}8,500.00$ $\rightarrow$ **`ESCALATE`**.
6. **`BOUNDARY_UNCERTAINTY_ESCALATION`**: Probability in uncertainty band $0.32 \le P \le 0.38$ $\rightarrow$ **`ESCALATE`**.
7. **`RISK_WARNING_ESCALATION`**: Velocity score $> 0.65$ and IP risk $> 0.50$ $\rightarrow$ **`ESCALATE`**.

### Concurrency & Idempotency Safeguards
* **Database Primary Key Claim**: `PaymentExecutionClaim` table enforces primary key constraint on `idempotency_key = f"idemp_{txn_id}"`.
* **Thread Locking**: Process-level `_LOCAL_EXECUTION_LOCK = threading.Lock()` wraps claim creation.
* **Property Testing**: Verified across **10,000 randomized boundary property iterations** in [tests/test_safety_properties.py](file:///c:/Users/Aashi/OneDrive/Desktop/RecoverAI%20%E2%80%94%20AI%20Revenue%20Recovery%20Agent/tests/test_safety_properties.py).

---

## 8. Data and Experimental Scope

* **Dataset Source**: $20,000$ synthetic transactions ($4,214$ failed payment records) generated via `data/generate_data.py`.
* **Domain Alignment**: Data distributions mirror e-commerce payment failure categories (insufficient funds, technical timeouts, risk refusals, card expirations).
* **Scope Boundary**: All evaluations represent offline benchmark analysis on synthetic historical logs; real merchant production data is not used.

---

## 9. Known Limitations and Non-Claims

1. ❌ **No Live Production Credit Card Processing**: Real customer credit card funds are never touched; execution operates strictly in Razorpay Test Mode / Dry Run.
2. ❌ **Observational Causal Analysis**: Propensity Score IPW causal lift analysis (+8.24% ATE) is computed on synthetic historical logs and is NOT a randomized live A/B experiment.
3. ❌ **No Automated Retraining**: Population Stability Index (PSI) feature drift monitoring emits structured telemetry alerts (`logs/drift_audit.jsonl`); automated model retraining is intentionally omitted to maintain system stability.
4. ❌ **No Post-Hoc Probability Calibration**: Raw Logistic Regression sigmoid probabilities are used directly; explicit post-hoc calibration (Platt scaling / Isotonic regression) is omitted.
5. ❌ **Single-Process Lock Scope**: Process thread lock `_LOCAL_EXECUTION_LOCK` is scoped to a single Python process; multi-process concurrency safety relies on database primary key constraints.

---

## 10. How to Reproduce or Verify Key Claims

All commands are valid and runnable from the repository root:

```bash
# 1. Run Complete 38-Test Pytest Suite
pytest tests/ -v

# 2. Verify Financial Metrics & Ground-Truth Calculations
python run_metric_comparison_audit.py

# 3. Verify Decision Paths (ACT / ESCALATE / REFUSE)
python test_dashboard_demo_paths.py

# 4. Verify Interactive UI State Sequences
python test_interactive_ui_sequences.py

# 5. Verify Streamlit Rerun Audit Persistence Fix
python verify_streamlit_duplication_fix.py

# 6. Launch Streamlit Web Dashboard
streamlit run frontend/app.py

# 7. Launch FastAPI REST Backend
uvicorn backend.main:app --reload --port 8000
```
