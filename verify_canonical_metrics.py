"""
RecoverAI — Standalone Canonical Financial Ground-Truth Verification Script.

Reads committed exp_0_baseline.joblib model artifact and data/raw/transactions.csv,
reproduces exact test split (seed=42, test_size=0.15, stratify=recovered),
and calculates ground-truth financial metrics directly using Pandas/NumPy.

PURE READ-ONLY WITH RESPECT TO REPOSITORY STATE.
"""

import os
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.features import add_engineered_features


# Configurable Action Cost Structure (INR)
ACTION_COSTS = {
    "retry": 5.0,
    "reminder": 2.0,
    "payment_link": 12.0,
    "no_action": 0.0
}


def assign_business_action(failure_reason: str):
    reason = str(failure_reason).lower()
    if reason in ["network_timeout", "technical_error"]:
        action = "retry"
    elif reason in ["insufficient_funds", "authentication_failed", "limit_exceeded"]:
        action = "payment_link"
    elif reason in ["bank_declined", "customer_cancelled"]:
        action = "reminder"
    else:
        action = "no_action"
    return action, ACTION_COSTS[action]


def main():
    model_path = os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib")
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    model = joblib.load(model_path)

    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    seed = 42
    idx_dev, idx_test, _, _ = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )
    test_df = failed_df.loc[idx_test].copy()

    exp0_features = ["amount", "hour", "day_of_week", "payment_method", "failure_reason"]
    test_probs = model.predict_proba(test_df[exp0_features])[:, 1]
    test_probs_series = pd.Series(test_probs, index=idx_test)

    # Label-aligned Series lookup
    aligned_probs = test_probs_series.loc[test_df.index].values
    amounts = test_df["amount"].values
    y_true = test_df["recovered"].astype(int).values
    reasons = test_df["failure_reason"].values

    actions_and_costs = [assign_business_action(r) for r in reasons]
    actions = np.array([ac[0] for ac in actions_and_costs])
    costs = np.array([ac[1] for ac in actions_and_costs])

    in_band_mask = (aligned_probs >= 0.35) & (aligned_probs < 0.50)
    count_in_band = int(np.sum(in_band_mask))

    # Evaluate Tau = 0.50
    mask_50 = (aligned_probs >= 0.50) & (actions != "no_action")
    sel_50 = int(np.sum(mask_50))
    rev_50 = float(np.sum(amounts[mask_50 & (y_true == 1)]))
    cost_50 = float(np.sum(costs[mask_50]))
    net_50 = rev_50 - cost_50

    # Evaluate Tau = 0.35
    mask_35 = (aligned_probs >= 0.35) & (actions != "no_action")
    sel_35 = int(np.sum(mask_35))
    rev_35 = float(np.sum(amounts[mask_35 & (y_true == 1)]))
    cost_35 = float(np.sum(costs[mask_35]))
    net_35 = rev_35 - cost_35

    gross_uplift = rev_35 - rev_50
    net_uplift = net_35 - net_50

    print("\n" + "=" * 70)
    print(" 💳 RECOVERAI — CANONICAL FINANCIAL GROUND-TRUTH VERIFICATION")
    print("=" * 70)
    print(f"Model Artifact Path : {model_path}")
    print(f"Model Artifact SHA  : {model_sha256}")
    print(f"Test Set Size (N)   : {len(test_df)} failed payment records")
    print(f"Probs in [0.35,0.50): {count_in_band}")
    print("-" * 70)
    print(f"DEFAULT POLICY (tau = 0.50):")
    print(f"  - Selected Payments   : {sel_50}")
    print(f"  - Realized Recovered : ₹{rev_50:,.2f}")
    print(f"  - Action Costs       : ₹{cost_50:,.2f}")
    print(f"  - Realized Net Value : ₹{net_50:,.2f}")
    print("-" * 70)
    print(f"EV-OPTIMIZED POLICY (tau = 0.35):")
    print(f"  - Selected Payments   : {sel_35}")
    print(f"  - Realized Recovered : ₹{rev_35:,.2f}")
    print(f"  - Action Costs       : ₹{cost_35:,.2f}")
    print(f"  - Realized Net Value : ₹{net_35:,.2f}")
    print("-" * 70)
    print(f"FINANCIAL UPLIFT METRICS:")
    print(f"  - Realized Gross Uplift : +₹{gross_uplift:,.2f}")
    print(f"  - Realized True Net Uplift: +₹{net_uplift:,.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
