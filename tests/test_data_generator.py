import os
import pytest
import pandas as pd
from data.generate_data import generate_pipeline, validate_dataset, PAYMENT_METHODS, FAILURE_MAPPING, RECOVERY_ACTIONS


def test_generator_runs_and_row_count():
    """1 & 2: Test dataset generates successfully and requested row count is respected."""
    df = generate_pipeline(rows=500, seed=123, output_dir="data/raw")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_unique_transaction_ids():
    """3: Test that transaction_id values are strictly unique."""
    df = generate_pipeline(rows=300, seed=456, output_dir="data/raw")
    assert df["transaction_id"].nunique() == 300


def test_required_columns_exist():
    """4: Test that all required schema columns exist."""
    df = generate_pipeline(rows=200, seed=789, output_dir="data/raw")
    expected_cols = [
        "transaction_id", "customer_id", "merchant_id", "amount", "currency",
        "payment_method", "payment_network", "payment_channel", "timestamp",
        "hour", "day_of_week", "is_weekend", "customer_age_days",
        "customer_previous_transactions", "customer_successful_transactions",
        "customer_historical_success_rate", "customer_lifetime_value",
        "customer_average_transaction", "customer_transaction_frequency_30d",
        "amount_vs_customer_average", "is_subscription", "subscription_age_days",
        "is_first_transaction", "previous_failures_24h", "previous_failures_7d",
        "previous_successes_30d", "transactions_24h", "transactions_7d",
        "device_changed", "location_changed", "ip_risk_score", "velocity_score",
        "payment_status", "failure_reason", "failure_category",
        "recovery_attempted", "recovery_action", "recovery_attempt_count",
        "customer_contacted_today", "recovery_success_probability",
        "recovered", "recovered_amount"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_valid_categorical_values():
    """5: Test that categorical columns only contain allowed valid values."""
    df = generate_pipeline(rows=300, seed=101, output_dir="data/raw")
    
    assert set(df["payment_method"]).issubset(set(PAYMENT_METHODS))
    assert set(df["payment_status"]).issubset({"success", "failed"})
    
    failed_df = df[df["payment_status"] == "failed"]
    assert set(failed_df["failure_reason"]).issubset(set(FAILURE_MAPPING.keys()))
    assert set(failed_df["failure_category"]).issubset(set(FAILURE_MAPPING.values()))
    assert set(failed_df["recovery_action"]).issubset(set(RECOVERY_ACTIONS))
    
    succ_df = df[df["payment_status"] == "success"]
    assert (succ_df["failure_reason"] == "none").all()
    assert (succ_df["failure_category"] == "none").all()


def test_target_values_validity():
    """6: Test target values validity (recovered, recovered_amount, bounds)."""
    df = generate_pipeline(rows=400, seed=202, output_dir="data/raw")
    
    # recovered is strictly 0 or 1
    assert set(df["recovered"].unique()).issubset({0, 1})
    
    # recovered_amount bounds
    assert (df["recovered_amount"] >= 0).all()
    assert (df["recovered_amount"] <= df["amount"] + 1e-4).all()
    
    # recovered_amount == 0 when recovered == 0
    not_recovered = df[df["recovered"] == 0]
    assert (not_recovered["recovered_amount"] == 0.0).all()


def test_no_target_leakage_in_features():
    """7: Test that target columns are not leaking into pre-decision input features."""
    df = generate_pipeline(rows=200, seed=303, output_dir="data/raw")
    
    # List of pure input features at transaction decision time
    input_features = [
        "amount", "payment_method", "payment_network", "payment_channel",
        "customer_age_days", "customer_previous_transactions",
        "customer_historical_success_rate", "customer_lifetime_value",
        "amount_vs_customer_average", "is_subscription",
        "previous_failures_24h", "previous_failures_7d",
        "ip_risk_score", "velocity_score", "failure_reason", "failure_category"
    ]
    
    # Target columns must be distinct from input features
    target_columns = ["recovered", "recovered_amount", "recovery_success_probability"]
    
    for feat in input_features:
        assert feat not in target_columns, f"Target leakage! {feat} found in target columns."


def test_validation_function_passes():
    """Test that validate_dataset returns True on generated dataset."""
    df = generate_pipeline(rows=300, seed=404, output_dir="data/raw")
    assert validate_dataset(df) is True
