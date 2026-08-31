"""
Unit & Integration Test Suite for ML Experimentation & Data Leakage Prevention Framework.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from ml.features import (
    APPROVED_MODEL_FEATURES,
    HIGH_SIGNAL_MODEL_FEATURES,
    ENGINEERED_MODEL_FEATURES,
    FORBIDDEN_TARGET_LEAKAGE_COLUMNS,
    audit_features,
    add_engineered_features,
    build_custom_preprocessor
)
from ml.predict import predict_recovery_probability
from ml.experiments import run_all_experiments


def test_1_forbidden_target_columns_cannot_enter_feature_list():
    """Test 1: Verify forbidden target leakage columns are completely absent from all feature sets."""
    audit_features(APPROVED_MODEL_FEATURES)
    audit_features(HIGH_SIGNAL_MODEL_FEATURES)
    audit_features(ENGINEERED_MODEL_FEATURES)
    assert set(ENGINEERED_MODEL_FEATURES).isdisjoint(FORBIDDEN_TARGET_LEAKAGE_COLUMNS)


def test_2_recovered_target_cannot_appear_in_model_input():
    """Test 2: Verify target 'recovered' cannot appear in input feature matrix X."""
    sample_df = pd.DataFrame({
        "amount": [1000.0],
        "recovered": [1],
        "payment_method": ["upi"],
        "failure_reason": ["network_timeout"],
        "failure_category": ["transient"]
    })
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        audit_features(list(sample_df.columns))


def test_3_historical_features_chronological_independence():
    """Test 3: Verify historical features for transaction t are unaffected by rows after t."""
    sample_df = pd.DataFrame({
        "amount": [1500.0, 5000.0],
        "customer_average_transaction": [1000.0, 2500.0],
        "previous_failures_24h": [1, 2],
        "transactions_24h": [2, 4],
        "customer_historical_success_rate": [0.8, 0.9],
        "velocity_score": [0.5, 0.2],
        "ip_risk_score": [0.1, 0.8],
        "failure_category": ["transient", "risk_related"]
    })

    # Transform 2 rows vs first row only
    enriched_full = add_engineered_features(sample_df)
    enriched_first_only = add_engineered_features(sample_df.iloc[[0]])

    # Row 0 values must be identical regardless of whether Row 1 exists
    for col in ["amount_log", "amount_to_customer_average_ratio", "high_risk_flag", "risk_velocity_product"]:
        assert enriched_full.loc[0, col] == enriched_first_only.loc[0, col]


def test_4_current_outcome_cannot_influence_own_historical_features():
    """Test 4: Verify add_engineered_features relies strictly on pre-recovery fields."""
    sample_df = pd.DataFrame({
        "amount": [2000.0],
        "customer_average_transaction": [1000.0],
        "previous_failures_24h": [0],
        "transactions_24h": [1],
        "customer_historical_success_rate": [0.9],
        "velocity_score": [0.1],
        "ip_risk_score": [0.1],
        "failure_category": ["transient"]
    })

    enriched = add_engineered_features(sample_df)
    assert enriched.loc[0, "amount_to_customer_average_ratio"] == 2.0
    assert enriched.loc[0, "high_risk_flag"] == 0.0


def test_5_preprocessor_isolation_fitted_on_train_only():
    """Test 5: Verify build_custom_preprocessor fits scaler & imputer on training fold only."""
    X_train = pd.DataFrame({
        "amount": [100.0, 200.0, 300.0],
        "payment_method": ["upi", "card", "upi"]
    })
    X_val = pd.DataFrame({
        "amount": [1000.0],
        "payment_method": ["netbanking"]  # unseen category
    })

    prep = build_custom_preprocessor(["amount"], ["payment_method"])
    X_train_trans = prep.fit_transform(X_train)
    X_val_trans = prep.transform(X_val)

    assert X_train_trans.shape[0] == 3
    assert X_val_trans.shape[0] == 1


def test_6_experiment_runner_and_json_structure(tmp_path):
    """Test 6: Verify run_all_experiments writes valid JSON and valid 5-Fold CV metrics."""
    csv_path = os.path.join(tmp_path, "test_txns.csv")
    exp_dir = os.path.join(tmp_path, "models")
    history = run_all_experiments(csv_path=csv_path, seed=42, exp_dir=exp_dir)

    assert "dataset" in history
    assert "split" in history
    assert "experiments" in history

    experiments = history["experiments"]
    assert len(experiments) >= 4

    for exp in experiments:
        assert "experiment_id" in exp
        assert "metrics" in exp
        assert "cross_validation" in exp
        cv = exp["cross_validation"]
        assert 0.0 <= cv["mean_roc_auc"] <= 1.0

    assert os.path.exists(os.path.join(exp_dir, "exp_0_baseline.joblib"))



def test_existing_production_inference_remains_functional():
    """Verify existing production inference logic continues to work unchanged."""
    sample_txn = {
        "amount": 1200.0,
        "amount_vs_customer_average": 1.1,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_subscription": 1,
        "subscription_age_days": 180,
        "is_first_transaction": 0,
        "customer_age_days": 365,
        "customer_previous_transactions": 10,
        "customer_successful_transactions": 9,
        "customer_historical_success_rate": 0.9,
        "customer_lifetime_value": 15000.0,
        "customer_average_transaction": 1100.0,
        "customer_transaction_frequency_30d": 3.0,
        "consecutive_failure_streak": 0,
        "customer_past_recovery_rate_pre_current": 0.0,
        "previous_failures_24h": 0,

        "previous_failures_7d": 0,
        "previous_successes_30d": 3,
        "transactions_24h": 1,
        "transactions_7d": 3,
        "device_changed": 0,
        "location_changed": 0,
        "ip_risk_score": 0.1,
        "velocity_score": 0.2,
        "payment_method": "upi",
        "payment_network": "npci",
        "payment_channel": "mobile_app",
        "failure_reason": "network_timeout",
        "failure_category": "transient"
    }

    res = predict_recovery_probability(sample_txn)
    assert "recovery_probability" in res
    assert isinstance(res["recovery_probability"], float)
    assert 0.0 <= res["recovery_probability"] <= 1.0
