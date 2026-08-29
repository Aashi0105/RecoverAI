"""
Regression test for probability Series index alignment in evaluate_threshold_grid.
Guarantees index=idx_test is preserved and prevents RangeIndex regression.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import add_engineered_features
from agent.orchestrator import load_orchestrator_model
from evaluation.business_metrics import evaluate_threshold_grid
from frontend.app import compute_live_top_metrics, load_and_split_dataset


def test_probability_index_alignment_regression():
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

    model = load_orchestrator_model()
    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    test_probs = model.predict_proba(test_df[exp0_features])[:, 1]

    # 1. Test correct index alignment
    correct_series = pd.Series(test_probs, index=idx_test)
    assert correct_series.index.equals(test_df.index), "Series index must equal test_df.index"

    m_def_correct = evaluate_threshold_grid(test_df, correct_series, thresholds=[0.50]).iloc[0].to_dict()
    m_opt_correct = evaluate_threshold_grid(test_df, correct_series, thresholds=[0.35]).iloc[0].to_dict()

    assert abs(m_def_correct["realized_recovered_revenue"] - 835012.80) < 0.01
    assert abs(m_opt_correct["realized_recovered_revenue"] - 900001.40) < 0.01
    assert abs((m_opt_correct["realized_recovered_revenue"] - m_def_correct["realized_recovered_revenue"]) - 64988.60) < 0.01

    # 2. Verify incorrect RangeIndex reproduces the historical bug
    range_series = pd.Series(test_probs)
    assert not range_series.index.equals(test_df.index), "RangeIndex should not equal test_df.index"

    m_def_wrong = evaluate_threshold_grid(test_df, range_series, thresholds=[0.50]).iloc[0].to_dict()
    assert abs(m_def_wrong["realized_recovered_revenue"] - 889907.57) < 0.01

    # 3. Verify Streamlit frontend function call
    metrics = compute_live_top_metrics(
        model_mtime=os.path.getmtime("ml/models/experiments/exp_0_baseline.joblib"),
        csv_mtime=os.path.getmtime(csv_path)
    )
    assert abs(metrics["default_recovered"] - 835012.80) < 0.01
    assert abs(metrics["opt_recovered"] - 900001.40) < 0.01
    assert abs(metrics["net_uplift_inr"] - 64988.60) < 0.01

    print("✅ REGRESSION TEST PASSED: Index alignment verified and RangeIndex bug prevented!")


if __name__ == "__main__":
    test_probability_index_alignment_regression()
