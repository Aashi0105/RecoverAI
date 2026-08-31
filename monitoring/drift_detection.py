"""
Population Stability Index (PSI) Feature Drift Detection Module for RecoverAI.

Computes baseline reference distributions on Development Data (N=3,581)
and measures statistical feature drift against inference/test batches.
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import add_engineered_features

import time

SNAPSHOT_PATH = os.path.join(PROJECT_ROOT, "monitoring", "baseline_snapshot.json")
DRIFT_AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "drift_audit.jsonl")

MONITORED_NUMERIC_FEATURES = ["amount", "hour", "day_of_week"]
MONITORED_CATEGORICAL_FEATURES = ["payment_method", "failure_reason"]


def get_drift_recommendation(status: str, feature_name: str) -> str:
    """
    Returns deterministic, actionable feature-level recommendations based on PSI status.
    """
    if status == "STABLE":
        return "No intervention required. Distribution is consistent with baseline."
    elif status == "MODERATE DRIFT":
        return f"Monitor feature '{feature_name}' distribution. Investigate upstream data pipeline if drift persists."
    else:  # SIGNIFICANT DRIFT
        return f"Investigate upstream data source/distribution for '{feature_name}'. Schedule model & data distribution review before retraining."


def append_drift_audit_event(summary: Dict[str, Any], log_path: str = DRIFT_AUDIT_LOG_PATH) -> None:
    """
    Appends structured drift audit event as JSON Line to log_path.
    Isolates file I/O exceptions to prevent disrupting monitoring computation.
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "system_status": summary.get("system_status", "UNKNOWN"),
            "significant_drift_count": summary.get("significant_drift_count", 0),
            "moderate_drift_count": summary.get("moderate_drift_count", 0),
            "monitored_features": summary.get("features", [])
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"⚠️ Warning: Failed to persist drift audit event: {e}")


def generate_baseline_snapshot(
    csv_path: str = "data/raw/transactions.csv",
    seed: int = 42,
    save_path: str = SNAPSHOT_PATH
) -> Dict[str, Any]:
    """
    Generates and saves baseline reference distribution on the Development Set (85%, N=3,581).
    Never uses or touches the untouched 15% test set.
    """
    full_csv_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
    df_raw = pd.read_csv(full_csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    idx_dev, _, _, _ = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )
    dev_df = failed_df.loc[idx_dev].copy()

    snapshot: Dict[str, Any] = {
        "metadata": {
            "dataset_source": csv_path,
            "dev_sample_size": len(dev_df),
            "seed": seed
        },
        "numeric": {},
        "categorical": {}
    }

    # 1. Numeric Features: 10 Quantile Bin Edges & Percentages
    for feat in MONITORED_NUMERIC_FEATURES:
        vals = dev_df[feat].dropna().values
        # Create 10 quantile bin edges
        quantiles = np.linspace(0.0, 1.0, 11)
        bin_edges = np.unique(np.quantile(vals, quantiles))
        
        # Ensure bin edges cover full range
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        counts, _ = np.histogram(vals, bins=bin_edges)
        total = len(vals)
        percentages = (counts / total).tolist()

        snapshot["numeric"][feat] = {
            "bin_edges": bin_edges.tolist(),
            "percentages": percentages,
            "total_count": total
        }

    # 2. Categorical Features: Value Counts & Percentages
    for feat in MONITORED_CATEGORICAL_FEATURES:
        counts = dev_df[feat].value_counts().to_dict()
        total = len(dev_df[feat].dropna())
        percentages = {k: v / total for k, v in counts.items()}

        snapshot["categorical"][feat] = {
            "categories": list(percentages.keys()),
            "percentages": percentages,
            "total_count": total
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot


def calculate_psi(
    baseline_pcts: List[float],
    current_pcts: List[float],
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))
    
    Status thresholds:
    - PSI < 0.10: STABLE
    - 0.10 <= PSI < 0.25: MODERATE DRIFT
    - PSI >= 0.25: SIGNIFICANT DRIFT
    """
    base = np.array(baseline_pcts, dtype=float)
    curr = np.array(current_pcts, dtype=float)

    # Epsilon smoothing to prevent division by zero or log(0)
    base = np.where(base <= 0, epsilon, base)
    curr = np.where(curr <= 0, epsilon, curr)

    # Re-normalize to unit sum
    base /= base.sum()
    curr /= curr.sum()

    psi_val = float(np.sum((curr - base) * np.log(curr / base)))

    if psi_val < 0.10:
        status = "STABLE"
    elif psi_val < 0.25:
        status = "MODERATE DRIFT"
    else:
        status = "SIGNIFICANT DRIFT"

    return round(psi_val, 4), status


def run_drift_report(
    current_df: pd.DataFrame,
    snapshot_path: str = SNAPSHOT_PATH,
    persist: bool = True,
    log_path: str = DRIFT_AUDIT_LOG_PATH
) -> Dict[str, Any]:
    """
    Evaluates current_df against frozen baseline_snapshot.json and outputs drift report.
    Persists audit event to log_path only when persist=True.
    """
    if not os.path.exists(snapshot_path):
        generate_baseline_snapshot(save_path=snapshot_path)

    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)

    report_features = []
    significant_drift_count = 0
    moderate_drift_count = 0

    # 1. Evaluate Numeric Features
    for feat, base_data in snapshot["numeric"].items():
        if feat not in current_df.columns:
            continue
        vals = current_df[feat].dropna().values
        bin_edges = np.array(base_data["bin_edges"])
        
        counts, _ = np.histogram(vals, bins=bin_edges)
        total = len(vals) if len(vals) > 0 else 1
        curr_pcts = (counts / total).tolist()

        psi_val, status = calculate_psi(base_data["percentages"], curr_pcts)

        if status == "SIGNIFICANT DRIFT":
            significant_drift_count += 1
        elif status == "MODERATE DRIFT":
            moderate_drift_count += 1

        rec = get_drift_recommendation(status, feat)

        report_features.append({
            "feature": feat,
            "type": "numeric",
            "psi": psi_val,
            "status": status,
            "recommendation": rec
        })

    # 2. Evaluate Categorical Features
    for feat, base_data in snapshot["categorical"].items():
        if feat not in current_df.columns:
            continue
        cats = base_data["categories"]
        base_pct_dict = base_data["percentages"]
        
        curr_counts = current_df[feat].value_counts().to_dict()
        total = len(current_df[feat].dropna()) if len(current_df[feat].dropna()) > 0 else 1

        # Align with baseline categories, grouping any unseen category into 'other' if needed
        base_pcts = []
        curr_pcts = []

        for c in cats:
            base_pcts.append(base_pct_dict.get(c, 0.0))
            curr_pcts.append(curr_counts.get(c, 0) / total)

        # Unseen categories handling
        unseen_count = sum(cnt for cat_name, cnt in curr_counts.items() if cat_name not in cats)
        if unseen_count > 0:
            base_pcts.append(0.0)
            curr_pcts.append(unseen_count / total)

        psi_val, status = calculate_psi(base_pcts, curr_pcts)

        if status == "SIGNIFICANT DRIFT":
            significant_drift_count += 1
        elif status == "MODERATE DRIFT":
            moderate_drift_count += 1

        rec = get_drift_recommendation(status, feat)

        report_features.append({
            "feature": feat,
            "type": "categorical",
            "psi": psi_val,
            "status": status,
            "recommendation": rec
        })

    overall_system_status = "HEALTHY"
    if significant_drift_count > 0:
        overall_system_status = f"REVIEW RECOMMENDED ({significant_drift_count} feature(s) showing SIGNIFICANT DRIFT)"
    elif moderate_drift_count > 0:
        overall_system_status = f"MONITORING ADVISED ({moderate_drift_count} feature(s) showing MODERATE DRIFT)"

    summary = {
        "features": report_features,
        "significant_drift_count": significant_drift_count,
        "moderate_drift_count": moderate_drift_count,
        "system_status": overall_system_status
    }

    # Persist structured drift audit event ONLY when persist=True
    if persist:
        append_drift_audit_event(summary, log_path=log_path)


    # Print clean report to console
    print("\n" + "=" * 65)
    print(" RECOVERAI FEATURE DRIFT REPORT (PSI AUDIT)")
    print("=" * 65)

    print(f"{'FEATURE':<20} | {'PSI SCORE':<10} | {'STATUS':<18} | {'RECOMMENDATION'}")
    print("-" * 65)
    for row in report_features:
        status_flag = row['status']
        print(f"{row['feature']:<20} | {row['psi']:<10.4f} | {status_flag:<18} | {row['recommendation']}")
    print("=" * 65)
    print(f"SYSTEM STATUS: {overall_system_status}")
    print("=" * 65 + "\n")

    return summary



if __name__ == "__main__":
    generate_baseline_snapshot()
    print("✅ Baseline snapshot generated successfully.")
