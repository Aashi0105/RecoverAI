"""
End-to-End Transaction Orchestrator for RecoverAI.

Connects:
1. EXP_0 Logistic Regression ML prediction
2. Coefficient-based Plain-English explainability
3. Frozen EV business policy (Tau = 0.35)
4. Deterministic safety decision engine (ACT / ESCALATE / REFUSE)
5. Policy-gated payment executor (Razorpay Test Mode / Dry Run)
6. Append-only JSONL audit trail (logs/decision_audit.jsonl)
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from ml.features import build_custom_preprocessor, add_engineered_features
from agent.nodes.policy import evaluate_transaction_policy, load_frozen_policy
from payment.executor import execute_recovery_policy

# Global Model Artifact Cache
_EXP0_PIPELINE = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ARTIFACT_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib"))
AUDIT_LOG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "logs", "decision_audit.jsonl"))


def load_orchestrator_model():
    """
    Lazy loads and caches EXP_0 Baseline Logistic Regression model.
    Falls back to fitting on Development data if model joblib is not on disk.
    """
    global _EXP0_PIPELINE
    if _EXP0_PIPELINE is not None:
        return _EXP0_PIPELINE

    if os.path.exists(MODEL_ARTIFACT_PATH):
        _EXP0_PIPELINE = joblib.load(MODEL_ARTIFACT_PATH)
        return _EXP0_PIPELINE

    # Fallback: Fit EXP_0 on Development set
    csv_path = "data/raw/transactions.csv"
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    seed = 42
    y_all = failed_df["recovered"].astype(int)
    idx_dev, _, _, _ = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )

    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    X_dev = failed_df.loc[idx_dev, exp0_num + exp0_cat]
    y_dev = y_all.loc[idx_dev]

    pipe = Pipeline([
        ("preprocessor", build_custom_preprocessor(exp0_num, exp0_cat)),
        ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
    ])
    pipe.fit(X_dev, y_dev)
    _EXP0_PIPELINE = pipe
    return _EXP0_PIPELINE


def generate_plain_english_explanation(
    pipeline,
    txn_df: pd.DataFrame,
    pred_prob: float
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates plain-English explanation using actual model coefficients and active feature values.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Transform single row input
    X_trans = preprocessor.transform(txn_df)[0]
    contributions = X_trans * coefficients

    positive_signals = []
    negative_signals = []

    for name, coef, val, contrib in zip(feature_names, coefficients, X_trans, contributions):
        if val == 0:
            continue

        clean_name = name.replace("num__", "").replace("cat__", "").replace("failure_reason_", "").replace("payment_method_", "")
        
        if contrib > 0.01:
            positive_signals.append({
                "feature": clean_name,
                "coefficient": round(float(coef), 4),
                "contribution": round(float(contrib), 4)
            })
        elif contrib < -0.01:
            negative_signals.append({
                "feature": clean_name,
                "coefficient": round(float(coef), 4),
                "contribution": round(float(contrib), 4)
            })

    # Sort signals by absolute contribution
    positive_signals.sort(key=lambda x: x["contribution"], reverse=True)
    negative_signals.sort(key=lambda x: x["contribution"])

    # Construct Plain-English Text
    reason = str(txn_df["failure_reason"].values[0])
    amount = float(txn_df["amount"].values[0])

    if pred_prob >= 0.60:
        pos_names = [s["feature"] for s in positive_signals[:2]]
        pos_str = ", ".join(pos_names) if pos_names else reason
        explanation = (
            f"Recovery probability is high ({pred_prob:.4f}) because this failure was classified as "
            f"'{reason}', which is historically a strong positive recovery signal. "
            f"Primary positive contributors include: {pos_str}."
        )
    elif pred_prob >= 0.35:
        explanation = (
            f"Recovery probability is moderate ({pred_prob:.4f}). Payment failure reason '{reason}' "
            f"provides sufficient recovery likelihood for economically viable intervention (Amount: ₹{amount:,.2f})."
        )
    else:
        neg_names = [s["feature"] for s in negative_signals[:2]]
        neg_str = ", ".join(neg_names) if neg_names else reason
        explanation = (
            f"Recovery probability is low ({pred_prob:.4f}) because '{reason}' is a strong "
            f"negative recovery predictor ({neg_str}). Autonomous recovery action is therefore not justified."
        )

    return explanation, positive_signals, negative_signals


def append_to_audit_log(audit_obj: Dict[str, Any], log_path: str = AUDIT_LOG_PATH) -> None:
    """
    Appends structured decision audit record as JSON Line to audit trail file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_obj) + "\n")


def orchestrate_transaction(
    transaction_input: Dict[str, Any],
    dry_run: bool = True,
    policy_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Single main interface executing complete end-to-end orchestration pipeline.
    """
    if policy_config is None:
        policy_config = load_frozen_policy()

    # 1. Format input DataFrame for EXP_0
    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    # Handle missing optional fields with standard fallbacks
    row_dict = {
        "amount": float(transaction_input.get("amount", 1000.0)),
        "hour": int(transaction_input.get("hour", 12)),
        "day_of_week": int(transaction_input.get("day_of_week", 2)),
        "payment_method": str(transaction_input.get("payment_method", "upi")),
        "failure_reason": str(transaction_input.get("failure_reason", "network_timeout"))
    }
    df_input = pd.DataFrame([row_dict])

    # 2. ML Prediction & Plain-English Explanation
    pipeline = load_orchestrator_model()
    pred_prob = float(pipeline.predict_proba(df_input[exp0_features])[0, 1])
    pred_prob_rounded = round(pred_prob, 4)

    explanation, pos_signals, neg_signals = generate_plain_english_explanation(
        pipeline, df_input, pred_prob_rounded
    )

    # 3. Deterministic Safety & Policy Evaluation
    policy_eval = evaluate_transaction_policy(transaction_input, pred_prob_rounded, policy_config)

    # 4. Policy-Gated Payment Execution
    exec_res = execute_recovery_policy(policy_eval, dry_run=dry_run)

    # 5. Assemble Complete JSON Audit Object
    audit_object = {
        "transaction_id": transaction_input.get("transaction_id", "txn_unknown"),
        "timestamp": transaction_input.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        "customer_id": transaction_input.get("customer_id", "cust_unknown"),
        "amount": round(float(transaction_input.get("amount", 0.0)), 2),
        
        "prediction": {
            "experiment_id": "EXP_0",
            "model_type": "Logistic Regression",
            "recovery_probability": pred_prob_rounded,
            "explanation": explanation,
            "top_positive_signals": pos_signals[:3],
            "top_negative_signals": neg_signals[:3]
        },

        "business_policy": {
            "threshold": policy_config.get("selected_threshold", 0.35),
            "recommended_action": policy_eval.get("recommended_action"),
            "action_cost": policy_eval.get("action_cost"),
            "expected_gross_recovery_value": round(pred_prob_rounded * float(transaction_input.get("amount", 0.0)), 2),
            "passed_ev_threshold": bool(pred_prob_rounded >= policy_config.get("selected_threshold", 0.35))
        },

        "safety": {
            "decision": policy_eval.get("decision"),
            "triggered_rules": policy_eval.get("triggered_rules", []),
            "justification": policy_eval.get("justification")
        },

        "execution": {
            "execution_status": exec_res.get("execution_status"),
            "external_api_called": exec_res.get("external_api_called"),
            "dry_run": exec_res.get("dry_run"),
            "reference_id": exec_res.get("razorpay_reference_id"),
            "short_url": exec_res.get("short_url"),
            "blocking_reason": exec_res.get("blocking_reason")
        },

        "audit_metadata": {
            "policy_version": policy_config.get("policy_version", "1.0"),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 6. Save to Append-Only JSONL Audit Log
    append_to_audit_log(audit_object)

    return audit_object
