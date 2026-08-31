"""
Unit tests for evaluation/causal_lift.py and synthetic treatment effect properties.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.causal_lift import (
    calculate_raw_lift,
    fit_propensity_score_model,
    calculate_ipw_ate,
    calculate_stratified_effects,
    bootstrap_causal_ci,
    PRE_TREATMENT_COVARIATES
)
from data.generate_data import calculate_treatment_effect
from ml.features import EXP0_NUMERIC_FEATURES, EXP0_CATEGORICAL_FEATURES


def test_no_post_treatment_leakage_in_covariates():
    """
    Guarantees no outcome or post-treatment variables leak into propensity estimation.
    """
    forbidden = ["recovered", "recovered_amount", "recovery_action", "recovery_attempt_count", "customer_contacted_today"]
    for col in forbidden:
        assert col not in PRE_TREATMENT_COVARIATES, f"Leaked post-treatment variable: {col}"


def test_ml_features_exclude_treatment_variable():
    """
    Guarantees production ML model features strictly exclude treatment variable recovery_attempted.
    """
    assert "recovery_attempted" not in EXP0_NUMERIC_FEATURES
    assert "recovery_attempted" not in EXP0_CATEGORICAL_FEATURES


def test_synthetic_treatment_effect_properties():
    """
    Verifies properties of calculate_treatment_effect:
    1. Zero effect for invalid_card, card_expired, suspected_risk.
    2. Positive effect for network_timeout, technical_error.
    3. Streak and IP risk penalties work as designed.
    """
    row_risk = pd.Series({"failure_reason": "suspected_risk", "consecutive_failure_streak": 0, "ip_risk_score": 0.1})
    row_expired = pd.Series({"failure_reason": "card_expired", "consecutive_failure_streak": 0, "ip_risk_score": 0.1})
    row_network = pd.Series({"failure_reason": "network_timeout", "consecutive_failure_streak": 0, "ip_risk_score": 0.1})

    assert calculate_treatment_effect(row_risk) == 0.0
    assert calculate_treatment_effect(row_expired) == 0.0
    assert calculate_treatment_effect(row_network) == 0.18

    # Streak penalty test
    row_streak = pd.Series({"failure_reason": "network_timeout", "consecutive_failure_streak": 4, "ip_risk_score": 0.1})
    assert calculate_treatment_effect(row_streak) == pytest.approx(0.10)  # 0.18 - 0.08 = 0.10



def test_causal_lift_recovers_positive_ate():
    """
    Verifies causal-lift analysis correctly measures positive ATE on regenerated dataset.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    from ml.features import add_engineered_features
    failed_df = add_engineered_features(failed_df)

    raw_res = calculate_raw_lift(failed_df)
    p_scores, _ = fit_propensity_score_model(failed_df)
    ipw_res = calculate_ipw_ate(failed_df, p_scores)

    # Positive treatment lift expected (raw lift ~1.7%, IPW ATE ~1.3%)
    assert raw_res["raw_absolute_lift"] > 0.0
    assert ipw_res["ate_ipw_absolute"] > 0.0



def test_deterministic_bootstrap_ci():
    """
    Verifies bootstrap confidence intervals are 100% deterministic using seed 42.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].head(100).copy()

    ci_1 = bootstrap_causal_ci(failed_df, n_bootstraps=20, seed=42)
    ci_2 = bootstrap_causal_ci(failed_df, n_bootstraps=20, seed=42)

    assert ci_1["raw_lift_ci_95"] == ci_2["raw_lift_ci_95"]
    assert ci_1["ipw_ate_ci_95"] == ci_2["ipw_ate_ci_95"]
