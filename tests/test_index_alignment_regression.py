"""
Regression test for probability Series index alignment in evaluate_threshold_grid.
Guarantees index=idx_test is preserved and prevents RangeIndex regression.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import add_engineered_features, APPROVED_MODEL_FEATURES
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
    test_probs = model.predict_proba(test_df[APPROVED_MODEL_FEATURES])[:, 1]


    # 1. Test correct index alignment
    correct_series = pd.Series(test_probs, index=idx_test)
    assert correct_series.index.equals(test_df.index), "Series index must equal test_df.index"

    m_def_correct = evaluate_threshold_grid(test_df, correct_series, thresholds=[0.50]).iloc[0].to_dict()
    m_opt_correct = evaluate_threshold_grid(test_df, correct_series, thresholds=[0.35]).iloc[0].to_dict()

    assert abs(m_def_correct["realized_recovered_revenue"] - 889907.57) < 0.01
    assert abs(m_opt_correct["realized_recovered_revenue"] - 920754.44) < 0.01
    assert abs((m_opt_correct["realized_recovered_revenue"] - m_def_correct["realized_recovered_revenue"]) - 30846.87) < 0.01

    # 2. Test reordered but alignable Series index
    shuffled_idx = idx_test[::-1]
    reordered_series = pd.Series(test_probs[::-1], index=shuffled_idx)
    m_reordered = evaluate_threshold_grid(test_df, reordered_series, thresholds=[0.50]).iloc[0].to_dict()
    assert abs(m_reordered["realized_recovered_revenue"] - 889907.57) < 0.01

    # 3. Verify unalignable RangeIndex Series raises ValueError under strict index-validation contract
    range_series = pd.Series(test_probs)
    assert not range_series.index.equals(test_df.index), "RangeIndex should not equal test_df.index"
    with pytest.raises(ValueError, match="oof_probs Series index does not contain all required df_dev indices"):
        evaluate_threshold_grid(test_df, range_series, thresholds=[0.50])

    # 4. Verify Streamlit frontend function call
    metrics = compute_live_top_metrics(
        model_mtime=os.path.getmtime("ml/models/experiments/exp_0_baseline.joblib"),
        csv_mtime=os.path.getmtime(csv_path)
    )
    assert abs(metrics["default_recovered"] - 889907.57) < 0.01
    assert abs(metrics["opt_recovered"] - 920754.44) < 0.01
    assert abs(metrics["gross_uplift_inr"] - 30846.87) < 0.01


    # 5. Verify Zero Intervention Baseline function
    from evaluation.business_metrics import evaluate_zero_intervention_baseline
    zero_metrics = evaluate_zero_intervention_baseline(test_df)
    assert zero_metrics["payments_selected"] == 0
    assert zero_metrics["action_costs_incurred"] == 0.0
    assert zero_metrics["realized_recovered_revenue"] == 0.0
    assert zero_metrics["realized_net_value"] == 0.0
    assert zero_metrics["natural_observed_recovered_revenue"] == 134391.02





    print("✅ REGRESSION TEST PASSED: Index alignment verified and RangeIndex bug prevented!")


if __name__ == "__main__":
    test_probability_index_alignment_regression()
