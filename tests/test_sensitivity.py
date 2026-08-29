"""
Unit tests for evaluation/sensitivity_analysis.py and business metric contract.
"""

import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.sensitivity_analysis import run_sensitivity_analysis


def test_sensitivity_analysis_grid_contract():
    """
    Verifies metric definitions, base case reproduction, and action cost propagation across 6 scenarios.
    """
    results_df = run_sensitivity_analysis()

    # 1. Base case verification
    base = results_df[results_df["Scenario"] == "1. BASE CASE"].iloc[0]
    assert base["Gross Uplift"] == "+₹64,988.60"
    assert base["True Net Uplift (₹)"] == "+₹64,359.60"
    assert abs(base["_raw_gross_uplift"] - 64988.60) < 0.01
    assert abs(base["_raw_true_net_uplift"] - 64359.60) < 0.01

    # 2. Action Cost +50% verification
    cost_plus = results_df[results_df["Scenario"] == "2. ACTION COST +50%"].iloc[0]
    assert cost_plus["Gross Uplift"] == "+₹64,988.60"  # Gross uplift unchanged
    assert cost_plus["True Net Uplift (₹)"] == "+₹64,045.10"  # True net uplift decreases by ₹314.50

    # 3. Action Cost -50% verification
    cost_minus = results_df[results_df["Scenario"] == "3. ACTION COST -50%"].iloc[0]
    assert cost_minus["Gross Uplift"] == "+₹64,988.60"  # Gross uplift unchanged
    assert cost_minus["True Net Uplift (₹)"] == "+₹64,674.10"  # True net uplift increases by ₹314.50

    # 4. Total Revenue at Risk constancy
    first_risk = results_df["Total Revenue at Risk"].iloc[0]
    for r in results_df["Total Revenue at Risk"]:
        assert r == first_risk, f"Total risk mismatch: {r} != {first_risk}"

    # 5. Positive True Net Uplift across all 6 scenarios
    for is_pos in results_df["Positive True Net Uplift?"]:
        assert is_pos == "Yes"
