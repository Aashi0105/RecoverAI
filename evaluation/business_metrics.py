"""
Business Expected Value (EV) Optimization & Financial Recovery Policy Engine.

Implements rigorous out-of-fold (OOF) decision policy optimization on development data,
followed by frozen single-pass evaluation on the untouched test set.
"""

import json
import os
import joblib

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from ml.features import build_custom_preprocessor, add_engineered_features


# Configurable Action Cost Structure (INR)
ACTION_COSTS: Dict[str, float] = {
    "retry": 5.0,
    "reminder": 2.0,
    "payment_link": 12.0,
    "no_action": 0.0
}


def assign_business_action(failure_reason: str) -> Tuple[str, float]:
    """
    Maps failure_reason to recommended recovery action and associated cost.
    Demonstrates domain action assignment based on gateway failure classification.
    """
    reason = str(failure_reason).lower()
    if reason in ["network_timeout", "technical_error"]:
        action = "retry"
    elif reason in ["insufficient_funds", "authentication_failed", "limit_exceeded"]:
        action = "payment_link"
    elif reason in ["bank_declined", "customer_cancelled"]:
        action = "reminder"
    else:
        # Card expired, invalid card, suspected risk -> zero ROI intervention
        action = "no_action"
    
    return action, ACTION_COSTS[action]


def compute_oof_predictions(
    failed_df: pd.DataFrame,
    dev_indices: pd.Index,
    seed: int = 42
) -> pd.Series:
    """
    Computes 5-Fold Stratified Out-of-Fold (OOF) predicted probabilities
    strictly on the Development set to prevent data leakage during optimization.
    """
    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    X_dev = failed_df.loc[dev_indices, exp0_features]
    y_dev = failed_df.loc[dev_indices, "recovered"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(dev_indices))

    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_va = X_dev.iloc[val_idx]

        pipe = Pipeline([
            ("preprocessor", build_custom_preprocessor(exp0_num, exp0_cat)),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
        ])
        pipe.fit(X_tr, y_tr)
        oof_probs[val_idx] = pipe.predict_proba(X_va)[:, 1]

    return pd.Series(oof_probs, index=dev_indices)


def evaluate_threshold_grid(
    df_dev: pd.DataFrame,
    oof_probs: pd.Series,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Evaluates Expected Value (EV) and operational metrics across candidate probability thresholds
    using out-of-fold predictions on development data.
    """
    if isinstance(oof_probs, pd.Series):
        if not oof_probs.index.equals(df_dev.index):
            if set(df_dev.index).issubset(set(oof_probs.index)):
                oof_probs = oof_probs.loc[df_dev.index]
            else:
                raise ValueError("oof_probs Series index does not contain all required df_dev indices")

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]


    y_true = df_dev["recovered"].astype(int).values
    amounts = df_dev["amount"].values
    reasons = df_dev["failure_reason"].values

    # Pre-assign action costs for each transaction
    actions_and_costs = [assign_business_action(r) for r in reasons]
    actions = [ac[0] for ac in actions_and_costs]
    costs = np.array([ac[1] for ac in actions_and_costs])

    total_samples = len(df_dev)
    total_revenue_potential = float(np.sum(amounts[y_true == 1]))

    records = []

    for tau in thresholds:
        # Policy rule: Intervene if predicted_prob >= tau AND assigned action != 'no_action'
        selected_mask = (oof_probs.values >= tau) & (np.array(actions) != "no_action")
        
        selected_count = int(np.sum(selected_mask))
        intervention_rate = (selected_count / total_samples * 100) if total_samples > 0 else 0.0

        # Action costs incurred
        action_costs_incurred = float(np.sum(costs[selected_mask]))

        # Expected Gross Recovery Revenue = sum(predicted_prob * amount for selected)
        expected_gross_val = float(np.sum(oof_probs.values[selected_mask] * amounts[selected_mask]))
        expected_net_val = expected_gross_val - action_costs_incurred

        # Realized Metrics (Out-of-sample ground truth)
        realized_recovered_val = float(np.sum(amounts[selected_mask & (y_true == 1)]))

        preds = selected_mask.astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)

        records.append({
            "threshold": tau,
            "payments_selected": selected_count,
            "intervention_rate_pct": round(intervention_rate, 2),
            "expected_gross_recovery_value": round(expected_gross_val, 2),
            "action_costs_incurred": round(action_costs_incurred, 2),
            "expected_net_value": round(expected_net_val, 2),
            "realized_recovered_revenue": round(realized_recovered_val, 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })

    return pd.DataFrame(records)


def evaluate_zero_intervention_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes structured zero-intervention operational baseline metrics for business evaluation.

    Operational Baseline Contract:
    - Zero automated interventions (0 payments selected, 0.00% intervention rate).
    - Automated action cost = ₹0.00.
    - Incremental recovered revenue & net value attributable to automated intervention = ₹0.00.
    - Observed natural recovery in test dataset is preserved separately for observational transparency.
    """
    total_samples = len(df)
    total_risk = float(np.sum(df["amount"])) if "amount" in df.columns else 0.0

    has_attempt = "recovery_attempted" in df.columns
    has_rec = "recovered" in df.columns

    if has_attempt and has_rec:
        nat_mask = (df["recovery_attempted"] == 0) & (df["recovered"] == 1)
        nat_count = int(np.sum(nat_mask))
        nat_rev = float(np.sum(df.loc[nat_mask, "amount"])) if "amount" in df.columns else 0.0
    else:
        nat_count = 0
        nat_rev = 0.0

    return {
        "strategy": "No automated intervention",
        "payments_selected": 0,
        "intervention_rate_pct": 0.0,
        "expected_gross_recovery_value": 0.0,
        "action_costs_incurred": 0.0,
        "expected_net_value": 0.0,
        "realized_recovered_revenue": 0.0,
        "realized_net_value": 0.0,
        "natural_observed_recovered_count": nat_count,
        "natural_observed_recovered_revenue": round(nat_rev, 2),
        "total_revenue_at_risk": round(total_risk, 2),
        "interpretation": "Operational baseline representing zero automated interventions and zero action costs."
    }



def run_business_optimization(
    csv_path: str = "data/raw/transactions.csv",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes formal 2-stage Business EV Optimization:
    1. Development OOF Optimization & Policy Freezing
    2. Single Final Untouched Test Set Evaluation
    """
    print("\n" + "=" * 75)
    print(" 💼 RECOVERAI BUSINESS EXPECTED VALUE (EV) OPTIMIZATION ENGINE")
    print("=" * 75)

    # 1. Load data & filter failed payments
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    # 2. Strict 85 / 15 Dev vs Test Split
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )

    print(f"📊 Dataset Split (Seed={seed}):")
    print(f"   - Dev Set  : {len(idx_dev):,} failed payment records (85%) [For OOF Optimization]")
    print(f"   - Test Set : {len(idx_test):,} failed payment records (15%) [UNTOUCHED Final Evaluation]")

    # 3. Compute Out-of-Fold (OOF) probabilities on Development Set
    print("\n⚡ Computing 5-Fold Stratified OOF Probabilities on Development Set...")
    oof_probs_dev = compute_oof_predictions(failed_df, idx_dev, seed=seed)

    # 4. Sweep candidate thresholds on Dev OOF
    df_dev = failed_df.loc[idx_dev].copy()
    grid_results = evaluate_threshold_grid(df_dev, oof_probs_dev)

    # 5. Policy Selection & Operational Constraint Analysis
    # Max Expected Net Value candidate
    max_ev_row = grid_results.loc[grid_results["expected_net_value"].idxmax()]
    
    # Balanced Operational Constraint Candidate (Tau >= 0.35 to prevent low-probability spamming)
    constrained_grid = grid_results[grid_results["threshold"] >= 0.35]
    balanced_row = constrained_grid.loc[constrained_grid["expected_net_value"].idxmax()]

    selected_policy_threshold = float(balanced_row["threshold"])
    print(f"\n🎯 Policy Optimization Results (Development Set):")
    print(f"   • Unconstrained Max EV Threshold : Tau = {max_ev_row['threshold']:.2f} (Net EV: ₹{max_ev_row['expected_net_value']:,.2f}, Intervene: {max_ev_row['intervention_rate_pct']}%)")
    print(f"   • Constrained Operational Winner  : Tau = {balanced_row['threshold']:.2f} (Net EV: ₹{balanced_row['expected_net_value']:,.2f}, Intervene: {balanced_row['intervention_rate_pct']}%)")

    # 6. Freeze Policy JSON
    policy_payload = {
        "policy_version": "1.0",
        "selection_dataset": "development_oof_5fold_cv",
        "objective": "maximize_expected_net_value_with_operational_constraint",
        "selected_threshold": selected_policy_threshold,
        "action_costs": ACTION_COSTS,
        "constraints": {
            "min_probability_threshold": 0.35,
            "max_allowed_action_cost_inr": 12.0
        },
        "assumptions": [
            "Action costs: retry=₹5.00, reminder=₹2.00, payment_link=₹12.00, no_action=₹0.00",
            "Expected Gross Value = P(recovered | features) * transaction_amount",
            "Expected Net Value = Expected Gross Value - Action Cost",
            "Probabilities are model-based expected value proxies, not causal uplift estimates."
        ],
        "created_from_experiment": "EXP_0"
    }

    os.makedirs("evaluation", exist_ok=True)
    policy_path = os.path.join("evaluation", "business_policy.json")
    with open(policy_path, "w") as f:
        json.dump(policy_payload, f, indent=2)
    print(f"💾 Policy frozen and saved to: {policy_path}")

    # 7. ONE Final Evaluation on Untouched Test Set
    print(f"\n❄️ Evaluating Frozen Policy (Tau = {selected_policy_threshold}) ONCE on untouched Test Set ($N = {len(idx_test)})...")
    
    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    X_dev_all = failed_df.loc[idx_dev, exp0_features]
    y_dev_all = failed_df.loc[idx_dev, "recovered"].astype(int)
    X_test_all = failed_df.loc[idx_test, exp0_features]
    y_test_all = failed_df.loc[idx_test, "recovered"].astype(int)

    # Load frozen EXP_0 pipeline artifact if available, otherwise fit on full Dev set
    model_path = os.path.join("ml", "models", "experiments", "exp_0_baseline.joblib")
    if os.path.exists(model_path):
        pipe_frozen = joblib.load(model_path)
    else:
        pipe_frozen = Pipeline([
            ("preprocessor", build_custom_preprocessor(exp0_num, exp0_cat)),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
        ])
        pipe_frozen.fit(X_dev_all, y_dev_all)


    test_probs = pipe_frozen.predict_proba(X_test_all)[:, 1]
    test_probs_series = pd.Series(test_probs, index=idx_test)

    df_test = failed_df.loc[idx_test].copy()

    # Benchmark 1: Default 0.50 Policy on Test Set
    test_default_df = evaluate_threshold_grid(df_test, test_probs_series, thresholds=[0.50])
    default_metrics = test_default_df.iloc[0].to_dict()

    # Benchmark 2: Frozen EV Optimized Policy on Test Set
    test_opt_df = evaluate_threshold_grid(df_test, test_probs_series, thresholds=[selected_policy_threshold])
    opt_metrics = test_opt_df.iloc[0].to_dict()

    print("\n" + "=" * 75)
    print(" 📈 FINAL UNTOUCHED TEST SET PERFORMANCE COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Default (Tau = 0.50)':<20} | {'EV Optimized (Tau = ' + str(selected_policy_threshold) + ')':<20}")
    print("-" * 75)
    print(f"{'Intervention Rate':<32} | {default_metrics['intervention_rate_pct']:>18.2f}% | {opt_metrics['intervention_rate_pct']:>18.2f}%")
    print(f"{'Payments Selected':<32} | {int(default_metrics['payments_selected']):>19d} | {int(opt_metrics['payments_selected']):>19d}")
    print(f"{'Expected Gross Recovery Value':<32} | ₹{default_metrics['expected_gross_recovery_value']:>17,.2f} | ₹{opt_metrics['expected_gross_recovery_value']:>17,.2f}")
    print(f"{'Action Costs Incurred':<32} | ₹{default_metrics['action_costs_incurred']:>17,.2f} | ₹{opt_metrics['action_costs_incurred']:>17,.2f}")
    print(f"{'Expected Net Value':<32} | ₹{default_metrics['expected_net_value']:>17,.2f} | ₹{opt_metrics['expected_net_value']:>17,.2f}")
    print(f"{'Realized Recovered Revenue':<32} | ₹{default_metrics['realized_recovered_revenue']:>17,.2f} | ₹{opt_metrics['realized_recovered_revenue']:>17,.2f}")
    print(f"{'Precision':<32} | {default_metrics['precision']:>19.4f} | {opt_metrics['precision']:>19.4f}")
    print(f"{'Recall':<32} | {default_metrics['recall']:>19.4f} | {opt_metrics['recall']:>19.4f}")
    print(f"{'F1 Score':<32} | {default_metrics['f1']:>19.4f} | {opt_metrics['f1']:>19.4f}")
    print("=" * 75 + "\n")

    # 8. Save Full Optimization Results JSON
    results_payload = {
        "development_grid_search": grid_results.to_dict(orient="records"),
        "test_set_evaluation": {
            "default_0_50_policy": default_metrics,
            "optimized_policy": opt_metrics
        }
    }

    results_path = os.path.join("evaluation", "business_optimization_results.json")
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"💾 Full optimization results saved to: {results_path}")

    return results_payload


def calculate_business_impact(
    amounts: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    default_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Compatibility function for offline training evaluation in ml/train.py.
    Calculates business financial impact metrics.
    """
    amounts_arr = np.asarray(amounts, dtype=float)
    y_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probabilities, dtype=float)

    total_count = len(amounts_arr)
    total_val = float(np.sum(amounts_arr))
    actual_recovered = float(np.sum(amounts_arr[y_arr == 1]))

    selected_mask = probs_arr >= default_threshold
    pred_recoverable = float(np.sum(amounts_arr[selected_mask]))
    expected_recovery = float(np.sum(probs_arr * amounts_arr))

    actual_selected_recovered = float(np.sum(amounts_arr[selected_mask & (y_arr == 1)]))
    rec_recall = (actual_selected_recovered / actual_recovered) if actual_recovered > 0 else 0.0
    rec_precision = (actual_selected_recovered / pred_recoverable) if pred_recoverable > 0 else 0.0

    return {
        "total_failed_count": total_count,
        "total_failed_value": total_val,
        "actual_recovered_revenue": actual_recovered,
        "predicted_recoverable_revenue": pred_recoverable,
        "expected_recovery_value": expected_recovery,
        "revenue_capture_recall": rec_recall,
        "revenue_precision": rec_precision
    }


def evaluate_decision_thresholds(
    amounts: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Compatibility function for threshold trade-off table in ml/train.py.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    amounts_arr = np.asarray(amounts, dtype=float)
    y_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probabilities, dtype=float)
    total_count = len(amounts_arr)
    actual_recovered_total = float(np.sum(amounts_arr[y_arr == 1]))

    records = []
    for tau in thresholds:
        selected_mask = probs_arr >= tau
        sel_count = int(np.sum(selected_mask))
        pct_sel = (sel_count / total_count * 100) if total_count > 0 else 0.0
        rev_sel = float(np.sum(amounts_arr[selected_mask]))
        act_rec = float(np.sum(amounts_arr[selected_mask & (y_arr == 1)]))
        rec_rate = (act_rec / rev_sel * 100) if rev_sel > 0 else 0.0
        missed_rev = actual_recovered_total - act_rec

        records.append({
            "threshold": tau,
            "payments_selected": sel_count,
            "percentage_selected": round(pct_sel, 2),
            "revenue_selected": round(rev_sel, 2),
            "actual_recovered_revenue": round(act_rec, 2),
            "recovery_rate_pct": round(rec_rate, 2),
            "missed_recoverable_revenue": round(missed_rev, 2)
        })

    return records


if __name__ == "__main__":
    run_business_optimization()
