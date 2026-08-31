"""
Feature Engineering and Data Preprocessing Pipeline for Payment Recovery Prediction.

Defines approved feature schema, strict target-leakage audits, and scikit-learn
ColumnTransformer pipelines.
"""

from typing import List, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Approved Feature Schema
NUMERIC_FEATURES = [
    "amount",
    "amount_vs_customer_average",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_subscription",
    "subscription_age_days",
    "is_first_transaction",
    "customer_age_days",
    "customer_previous_transactions",
    "customer_successful_transactions",
    "customer_historical_success_rate",
    "customer_lifetime_value",
    "customer_average_transaction",
    "customer_transaction_frequency_30d",
    "consecutive_failure_streak",
    "customer_past_recovery_rate_pre_current",
    "previous_failures_24h",
    "previous_failures_7d",
    "previous_successes_30d",
    "transactions_24h",
    "transactions_7d",
    "device_changed",
    "location_changed",
    "ip_risk_score",
    "velocity_score"
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "payment_network",
    "payment_channel",
    "failure_reason",
    "failure_category"
]

APPROVED_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# EXP_0 Baseline Model Feature Constants
EXP0_NUMERIC_FEATURES = ["amount", "hour", "day_of_week"]
EXP0_CATEGORICAL_FEATURES = ["payment_method", "failure_reason"]


# High-Signal Feature Selection (Purging 15 uninformative noise features)
HIGH_SIGNAL_NUMERIC_FEATURES = [
    "amount",
    "amount_vs_customer_average",
    "customer_historical_success_rate",
    "customer_previous_transactions",
    "previous_failures_24h",
    "previous_failures_7d",
    "ip_risk_score",
    "velocity_score",
    "consecutive_failure_streak",
    "customer_past_recovery_rate_pre_current"
]
HIGH_SIGNAL_CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_reason",
    "failure_category"
]
HIGH_SIGNAL_MODEL_FEATURES = HIGH_SIGNAL_NUMERIC_FEATURES + HIGH_SIGNAL_CATEGORICAL_FEATURES

# Engineered Features (Calculated strictly BEFORE recovery outcome)
ENGINEERED_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "amount_log",
    "amount_to_customer_average_ratio",
    "failure_frequency_ratio",
    "risk_velocity_product",
    "success_rate_velocity_interaction",
    "high_risk_flag",
    "failure_burst_flag",
    "failure_severity_score"
]
ENGINEERED_MODEL_FEATURES = ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 2. Forbidden Target Leakage Columns
FORBIDDEN_TARGET_LEAKAGE_COLUMNS = {
    "recovered",
    "recovered_amount",
    "recovery_success_probability",
    "recovery_attempted",
    "recovery_action",
    "recovery_attempt_count",
    "customer_contacted_today",
    "payment_status"
}

# Domain Severity Mapping for Failure Categories (Non-leaking ordinal mapping)
FAILURE_SEVERITY_MAP = {
    "transient": 1.0,
    "technical": 2.0,
    "customer_action_required": 3.0,
    "bank_decline": 4.0,
    "payment_method_problem": 5.0,
    "risk_related": 6.0,
    "none": 0.0
}


def _get_series(df: pd.DataFrame, col: str, default_val: Any) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default_val)
    return pd.Series(default_val, index=df.index)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely adds non-leaking engineered features to dataframe prior to feature extraction.
    Zero leakage: All inputs are known prior to payment recovery decision.
    """
    df_out = df.copy()

    
    # 1. Log-transformed amount
    amount_s = _get_series(df_out, "amount", 0.0)
    df_out["amount_log"] = np.log1p(np.maximum(amount_s.values, 0.0))
    
    # 2. Ratio of transaction amount to customer average transaction
    avg_txn_s = _get_series(df_out, "customer_average_transaction", 1.0)
    avg_txn = np.maximum(avg_txn_s.values, 1.0)
    df_out["amount_to_customer_average_ratio"] = amount_s.values / avg_txn

    # 3. Failure frequency ratio in last 24h
    txns_24h_s = _get_series(df_out, "transactions_24h", 1.0)
    txns_24h = np.maximum(txns_24h_s.values, 1.0)
    fails_24h_s = _get_series(df_out, "previous_failures_24h", 0.0)
    df_out["failure_frequency_ratio"] = fails_24h_s.values / txns_24h

    # 4. Risk-Velocity Product
    ip_risk = _get_series(df_out, "ip_risk_score", 0.0).values
    velocity = _get_series(df_out, "velocity_score", 0.0).values
    df_out["risk_velocity_product"] = ip_risk * velocity

    # 5. Success rate & velocity interaction term
    succ_rate = _get_series(df_out, "customer_historical_success_rate", 0.5).values
    df_out["success_rate_velocity_interaction"] = succ_rate * velocity

    # 6. High risk flag
    df_out["high_risk_flag"] = ((ip_risk > 0.65) | (velocity > 0.65)).astype(float)

    # 7. Failure burst flag
    fails_7d_s = _get_series(df_out, "previous_failures_7d", 0.0)
    df_out["failure_burst_flag"] = ((fails_24h_s.values >= 2) | (fails_7d_s.values >= 3)).astype(float)

    # 8. Failure severity score
    cats = _get_series(df_out, "failure_category", "none").astype(str)
    df_out["failure_severity_score"] = cats.map(lambda c: FAILURE_SEVERITY_MAP.get(c, 3.0)).values

    return df_out


engineer_features = add_engineered_features



def audit_features(feature_list: List[str]) -> None:
    """
    Data Leakage Audit: Raises ValueError if any target or post-recovery outcome field
    is detected in the feature set.
    """
    detected_leakage = set(feature_list).intersection(FORBIDDEN_TARGET_LEAKAGE_COLUMNS)
    if detected_leakage:
        raise ValueError(
            f"❌ CRITICAL TARGET LEAKAGE DETECTED! The following post-recovery or target fields "
            f"were found in the input features: {list(detected_leakage)}"
        )


def build_custom_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for dynamic feature subsets.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for default 29 numerical and categorical features.
    """
    return build_custom_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate only approved model features from a raw/processed dataframe.
    """
    # Audit for target leakage
    audit_features(list(df.columns))
    
    # Filter to approved features present in df
    missing_feats = [f for f in APPROVED_MODEL_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}")
        
    return df[APPROVED_MODEL_FEATURES].copy()


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filters dataset to failed payments only, audits feature list, and returns (X, y).
    """
    # Filter only failed payments
    if "payment_status" in df.columns:
        failed_df = df[df["payment_status"] == "failed"].copy()
    else:
        failed_df = df.copy()

    if len(failed_df) == 0:
        raise ValueError("No failed payments found in the dataset for model training!")

    if "recovered" not in failed_df.columns:
        raise ValueError("Target column 'recovered' not found in dataframe!")

    y = failed_df["recovered"].astype(int)
    X = failed_df[APPROVED_MODEL_FEATURES].copy()

    # Perform strict data leakage audit on X
    audit_features(list(X.columns))

    return X, y
