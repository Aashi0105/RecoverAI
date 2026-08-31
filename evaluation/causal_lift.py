"""
Causal-Lift & Treatment-Effect Evaluation Engine for RecoverAI.

Calculates Observational Association, Propensity Score Inverse Probability Weighting (IPW),
Stratified Subgroup Treatment Effects, and 95% Bootstrap Confidence Intervals for recovery interventions.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import add_engineered_features

PRE_TREATMENT_NUMERIC = [
    "amount", "hour", "day_of_week", "consecutive_failure_streak",
    "customer_past_recovery_rate_pre_current", "ip_risk_score", "velocity_score"
]
PRE_TREATMENT_CATEGORICAL = ["payment_method", "failure_reason"]
PRE_TREATMENT_COVARIATES = PRE_TREATMENT_NUMERIC + PRE_TREATMENT_CATEGORICAL


def calculate_raw_lift(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes raw observational recovery rates and unadjusted lift.
    """
    treated_mask = df["recovery_attempted"] == 1
    control_mask = df["recovery_attempted"] == 0

    n_treated = int(np.sum(treated_mask))
    n_control = int(np.sum(control_mask))

    p_treated = float(df.loc[treated_mask, "recovered"].mean()) if n_treated > 0 else 0.0
    p_control = float(df.loc[control_mask, "recovered"].mean()) if n_control > 0 else 0.0

    abs_lift = p_treated - p_control
    rel_lift = (abs_lift / p_control * 100) if p_control > 0 else 0.0

    return {
        "n_treated": n_treated,
        "n_control": n_control,
        "p_treated": p_treated,
        "p_control": p_control,
        "raw_absolute_lift": abs_lift,
        "raw_relative_lift_pct": rel_lift
    }


def fit_propensity_score_model(df: pd.DataFrame) -> Tuple[np.ndarray, Pipeline]:
    """
    Fits a Propensity Score Logistic Regression model predicting treatment T = recovery_attempted
    using strictly pre-treatment covariates.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), PRE_TREATMENT_NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), PRE_TREATMENT_CATEGORICAL)
        ]
    )

    propensity_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]
    )

    X = df[PRE_TREATMENT_COVARIATES]
    y_treatment = df["recovery_attempted"].astype(int)

    propensity_pipeline.fit(X, y_treatment)
    propensity_scores = propensity_pipeline.predict_proba(X)[:, 1]

    # Clip propensity scores to prevent extreme weights (positivity condition)
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

    return propensity_scores, propensity_pipeline


def calculate_ipw_ate(df: pd.DataFrame, propensity_scores: np.ndarray) -> Dict[str, Any]:
    """
    Computes Inverse Probability Weighting (IPW) Average Treatment Effect (ATE).
    ATE = E [ (T * Y) / e(X) - ((1 - T) * Y) / (1 - e(X)) ]
    """
    T = df["recovery_attempted"].astype(int).values
    Y = df["recovered"].astype(int).values
    e = propensity_scores

    # Horvitz-Thompson / Hajek IPW estimator
    w_t = T / e
    w_c = (1 - T) / (1 - e)

    mu_1 = np.sum(w_t * Y) / np.sum(w_t)
    mu_0 = np.sum(w_c * Y) / np.sum(w_c)

    ate = float(mu_1 - mu_0)
    rel_ate_pct = float((ate / mu_0 * 100)) if mu_0 > 0 else 0.0

    return {
        "adjusted_mu_treated": float(mu_1),
        "adjusted_mu_control": float(mu_0),
        "ate_ipw_absolute": ate,
        "ate_ipw_relative_pct": rel_ate_pct,
        "min_propensity": float(np.min(e)),
        "max_propensity": float(np.max(e)),
        "mean_propensity": float(np.mean(e))
    }


def calculate_stratified_effects(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """
    Computes subgroup treatment effects across categories (e.g., failure_reason, payment_method).
    """
    results = []
    for val, group in df.groupby(stratum_col):
        t_mask = group["recovery_attempted"] == 1
        c_mask = group["recovery_attempted"] == 0

        n_t = int(np.sum(t_mask))
        n_c = int(np.sum(c_mask))

        p_t = float(group.loc[t_mask, "recovered"].mean()) if n_t > 0 else 0.0
        p_c = float(group.loc[c_mask, "recovered"].mean()) if n_c > 0 else 0.0

        lift = p_t - p_c
        is_small = (n_t < 30 or n_c < 30)

        results.append({
            "stratum_column": stratum_col,
            "stratum_value": str(val),
            "n_treated": n_t,
            "n_control": n_c,
            "p_treated": round(p_t, 4),
            "p_control": round(p_c, 4),
            "absolute_lift": round(lift, 4),
            "is_small_sample": is_small
        })

    return pd.DataFrame(results)


def bootstrap_causal_ci(
    df: pd.DataFrame,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes non-parametric 95% Bootstrap Confidence Intervals for raw lift and IPW ATE.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(df)

    raw_lifts = []
    ipw_ates = []
    failed_iterations = 0

    for _ in range(n_bootstraps):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_df = df.iloc[sample_idx].copy()

        # Raw lift
        raw_res = calculate_raw_lift(boot_df)
        raw_lifts.append(raw_res["raw_absolute_lift"])

        # IPW ATE
        try:
            p_scores, _ = fit_propensity_score_model(boot_df)
            ipw_res = calculate_ipw_ate(boot_df, p_scores)
            ipw_ates.append(ipw_res["ate_ipw_absolute"])
        except Exception:
            failed_iterations += 1

    if failed_iterations > 0 and failed_iterations > int(n_bootstraps * 0.20):
        import warnings
        warnings.warn(
            f"High number of failed IPW bootstrap iterations ({failed_iterations}/{n_bootstraps}). "
            "Confidence interval may have reduced sample coverage.",
            RuntimeWarning
        )

    alpha = (1.0 - confidence_level) / 2.0
    raw_ci = (
        float(np.percentile(raw_lifts, alpha * 100)),
        float(np.percentile(raw_lifts, (1.0 - alpha) * 100))
    )

    if ipw_ates:
        ipw_ci = (
            float(np.percentile(ipw_ates, alpha * 100)),
            float(np.percentile(ipw_ates, (1.0 - alpha) * 100))
        )
    else:
        ipw_ci = (np.nan, np.nan)

    return {
        "raw_lift_ci_95": raw_ci,
        "ipw_ate_ci_95": ipw_ci
    }


def run_full_causal_analysis(csv_path: str = None) -> Dict[str, Any]:
    """
    Main entrypoint running the complete causal-lift and treatment-effect evaluation.
    """
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")

    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)

    # 1. Raw Observational Comparison
    raw_results = calculate_raw_lift(failed_df)

    # 2. Propensity Score Model & IPW ATE
    p_scores, p_pipeline = fit_propensity_score_model(failed_df)
    ipw_results = calculate_ipw_ate(failed_df, p_scores)

    # 3. Stratified Subgroup Analysis
    reason_strata = calculate_stratified_effects(failed_df, "failure_reason")
    method_strata = calculate_stratified_effects(failed_df, "payment_method")

    # 4. Bootstrap Confidence Intervals (Fast 200 iterations for default run)
    ci_results = bootstrap_causal_ci(failed_df, n_bootstraps=200, seed=42)

    return {
        "raw_metrics": raw_results,
        "ipw_metrics": ipw_results,
        "ci_metrics": ci_results,
        "failure_reason_strata": reason_strata,
        "payment_method_strata": method_strata
    }


if __name__ == "__main__":
    print("=" * 85)
    print(" 🔬 RECOVERAI CAUSAL-LIFT & TREATMENT-EFFECT ANALYSIS MODULE")
    print("=" * 85)

    results = run_full_causal_analysis()

    raw = results["raw_metrics"]
    ipw = results["ipw_metrics"]
    ci = results["ci_metrics"]

    print("\n1. RAW OBSERVATIONAL COMPARISON:")
    print(f" • Treated (T=1) Count     : {raw['n_treated']:,} (Recovery Rate: {raw['p_treated']*100:.2f}%)")
    print(f" • Control (T=0) Count     : {raw['n_control']:,} (Recovery Rate: {raw['p_control']*100:.2f}%)")
    print(f" • Raw Absolute Lift       : {raw['raw_absolute_lift']*100:+.2f} percentage points")
    print(f" • Raw 95% CI              : [{ci['raw_lift_ci_95'][0]*100:+.2f}%, {ci['raw_lift_ci_95'][1]*100:+.2f}%]")

    print("\n2. PROPENSITY-SCORE IPW ADJUSTED TREATMENT EFFECT (ATE):")
    print(f" • Adjusted Treated Mean   : {ipw['adjusted_mu_treated']*100:.2f}%")
    print(f" • Adjusted Control Mean   : {ipw['adjusted_mu_control']*100:.2f}%")
    print(f" • IPW Adjusted ATE        : {ipw['ate_ipw_absolute']*100:+.2f} percentage points")
    print(f" • IPW ATE 95% CI          : [{ci['ipw_ate_ci_95'][0]*100:+.2f}%, {ci['ipw_ate_ci_95'][1]*100:+.2f}%]")
    print(f" • Propensity Range        : [{ipw['min_propensity']:.4f}, {ipw['max_propensity']:.4f}] (Mean: {ipw['mean_propensity']:.4f})")

    print("\n3. STRATIFIED FAILURE REASON LIFT (SAMPLE):")
    print(results["failure_reason_strata"].to_string(index=False))

    print("\n=" * 85 + "\n")
