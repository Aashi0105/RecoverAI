"""
Sensitivity Analysis Module for RecoverAI Business Policy Engine.

Evaluates financial recovery performance under 6 scenario stress tests on the untouched test set (N=633):
1. BASE CASE: Current assumptions (reproduces ₹64,988.60 exactly).
2. ACTION COST +50%: 1.5x action cost multiplier.
3. ACTION COST -50%: 0.5x action cost multiplier.
4. THRESHOLD +0.05: tau = 0.40 EV threshold.
5. THRESHOLD -0.05: tau = 0.30 EV threshold.
6. PROBABILITY MISCALIBRATION: Shift predicted probabilities by -0.05.
"""

import os
import sys
from typing import Dict, Any, List
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import add_engineered_features, EXP0_NUMERIC_FEATURES, EXP0_CATEGORICAL_FEATURES
from agent.orchestrator import load_orchestrator_model
from evaluation.business_metrics import evaluate_threshold_grid, ACTION_COSTS


def run_sensitivity_analysis() -> pd.DataFrame:
    """
    Reruns the existing uplift calculation across 6 scenario stress tests.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    seed = 42
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )
    test_df = failed_df.loc[idx_test].copy()

    pipeline = load_orchestrator_model()
    exp0_features = EXP0_NUMERIC_FEATURES + EXP0_CATEGORICAL_FEATURES
    base_probs = pipeline.predict_proba(test_df[exp0_features])[:, 1]



    expected_total_risk = float(np.sum(test_df["amount"]))

    scenarios: List[Dict[str, Any]] = [
        {
            "name": "1. BASE CASE",
            "cost_mult": 1.0,
            "tau_def": 0.50,
            "tau_opt": 0.35,
            "prob_shift": 0.0
        },
        {
            "name": "2. ACTION COST +50%",
            "cost_mult": 1.50,
            "tau_def": 0.50,
            "tau_opt": 0.35,
            "prob_shift": 0.0
        },
        {
            "name": "3. ACTION COST -50%",
            "cost_mult": 0.50,
            "tau_def": 0.50,
            "tau_opt": 0.35,
            "prob_shift": 0.0
        },
        {
            "name": "4. THRESHOLD +0.05 (tau=0.40)",
            "cost_mult": 1.0,
            "tau_def": 0.50,
            "tau_opt": 0.40,
            "prob_shift": 0.0
        },
        {
            "name": "5. THRESHOLD -0.05 (tau=0.30)",
            "cost_mult": 1.0,
            "tau_def": 0.50,
            "tau_opt": 0.30,
            "prob_shift": 0.0
        },
        {
            "name": "6. PROBABILITY MISCALIBRATION (-0.05)",
            "cost_mult": 1.0,
            "tau_def": 0.50,
            "tau_opt": 0.35,
            "prob_shift": -0.05
        }
    ]

    results = []

    for sc in scenarios:
        current_costs = {k: v * sc["cost_mult"] for k, v in ACTION_COSTS.items()}
        current_probs = np.clip(base_probs + sc["prob_shift"], 0.0, 1.0)
        probs_series = pd.Series(current_probs, index=idx_test)

        with patch.dict("evaluation.business_metrics.ACTION_COSTS", current_costs):
            m_def = evaluate_threshold_grid(test_df, probs_series, thresholds=[sc["tau_def"]]).iloc[0].to_dict()
            m_opt = evaluate_threshold_grid(test_df, probs_series, thresholds=[sc["tau_opt"]]).iloc[0].to_dict()

        total_risk = float(np.sum(test_df["amount"]))
        if abs(total_risk - expected_total_risk) > 0.01:
            print(f"⚠️ BUG WARNING: Total revenue at risk changed in scenario '{sc['name']}'! Got ₹{total_risk}")

        def_gross = m_def["realized_recovered_revenue"]
        def_cost = m_def["action_costs_incurred"]
        def_net = def_gross - def_cost

        opt_gross = m_opt["realized_recovered_revenue"]
        opt_cost = m_opt["action_costs_incurred"]
        opt_net = opt_gross - opt_cost

        gross_uplift = opt_gross - def_gross
        true_net_uplift = opt_net - def_net
        true_pct_uplift = (true_net_uplift / total_risk * 100) if total_risk > 0 else 0.0
        is_positive = "Yes" if true_net_uplift > 0 else "No"

        results.append({
            "Scenario": sc["name"],
            "Total Revenue at Risk": f"₹{total_risk:,.2f}",
            "Default Gross Revenue": f"₹{def_gross:,.2f}",
            "Default Action Costs": f"₹{def_cost:,.2f}",
            "Default Net Value": f"₹{def_net:,.2f}",
            "EV Gross Revenue": f"₹{opt_gross:,.2f}",
            "EV Action Costs": f"₹{opt_cost:,.2f}",
            "EV Net Value": f"₹{opt_net:,.2f}",
            "Gross Uplift": f"{'+' if gross_uplift >= 0 else '-'}\u20b9{abs(gross_uplift):,.2f}",
            "True Net Uplift (\u20b9)": f"{'+' if true_net_uplift >= 0 else '-'}\u20b9{abs(true_net_uplift):,.2f}",

            "True Net Uplift (%)": f"{true_pct_uplift:+.2f}%",
            "Positive True Net Uplift?": is_positive,
            "_raw_true_net_uplift": true_net_uplift,
            "_raw_gross_uplift": gross_uplift
        })

    results_df = pd.DataFrame(results)

    # Save clean CSV artifact
    csv_out_path = os.path.join(PROJECT_ROOT, "evaluation", "sensitivity_results.csv")
    csv_export_df = results_df.drop(columns=["_raw_true_net_uplift", "_raw_gross_uplift"])
    csv_export_df.to_csv(csv_out_path, index=False)

    return results_df


def main():
    print("=" * 120)
    print(" 🔬 RECOVERAI SENSITIVITY ANALYSIS (6 SCENARIO STRESS TEST GRID)")
    print("=" * 120)

    df_res = run_sensitivity_analysis()

    display_df = df_res.drop(columns=["_raw_true_net_uplift", "_raw_gross_uplift"])
    print(display_df.to_string(index=False))

    positive_count = sum(1 for x in df_res["_raw_true_net_uplift"] if x > 0)
    total_count = len(df_res)
    min_uplift = df_res["_raw_true_net_uplift"].min()
    max_uplift = df_res["_raw_true_net_uplift"].max()

    base_row = df_res[df_res["Scenario"] == "1. BASE CASE"].iloc[0]

    summary_statement = (
        f"\nThe optimized policy produced a positive true net uplift in {positive_count} of {total_count} tested scenarios, "
        f"with true net uplift ranging from ₹{min_uplift:,.2f} to ₹{max_uplift:,.2f} "
        f"(base case gross uplift: ₹{base_row['_raw_gross_uplift']:,.2f}; base case true net uplift after incremental action costs: ₹{base_row['_raw_true_net_uplift']:,.2f})."
    )

    print("\n" + "=" * 120)
    print(" 📝 SUMMARY STATEMENT:")
    print(summary_statement)
    print("=" * 120 + "\n")



if __name__ == "__main__":
    main()
