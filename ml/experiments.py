"""
ML Experiment Tracking and Comparison System for RecoverAI.

Runs progressive, reproducible experiments with 5-Fold Stratified Cross-Validation:
- EXP_0: Baseline Logistic Regression (5 basic features)
- EXP_1: Full-Feature Random Forest (29 raw features)
- EXP_2: High-Signal Feature Selection (13 core features)
- EXP_3: Advanced Engineered Features (34 features)
- EXP_4: Tuned XGBoost Candidate (Tuned via RandomizedSearchCV on train data)
- EXP_5: Calibrated Classifier Candidate (Platt / Isotonic Calibration)

Saves artifacts separately in ml/models/experiments/ without overwriting production model.
Evaluates model selection via 5-Fold Stratified CV on Development data, then evaluates final candidate ONCE on untouched Test set.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from data.generate_data import generate_pipeline
from ml.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    APPROVED_MODEL_FEATURES,
    HIGH_SIGNAL_NUMERIC_FEATURES,
    HIGH_SIGNAL_CATEGORICAL_FEATURES,
    HIGH_SIGNAL_MODEL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    ENGINEERED_MODEL_FEATURES,
    audit_features,
    add_engineered_features,
    build_custom_preprocessor
)
from evaluation.model_metrics import compute_all_metrics


def run_cross_validation(X: pd.DataFrame, y: pd.Series, numeric_cols: List[str], categorical_cols: List[str], model_factory, seed: int = 42, n_splits: int = 5) -> Dict[str, Any]:
    """
    Executes Stratified 5-Fold Cross-Validation on training/development set.
    Fits preprocessor EXCLUSIVELY on each training fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs, fold_pr_aucs, fold_f1s = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        X_fold_val, y_fold_val = X.iloc[val_idx], y.iloc[val_idx]

        prep = build_custom_preprocessor(numeric_cols, categorical_cols)
        clf = model_factory()

        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
        pipe.fit(X_fold_train, y_fold_train)

        probs = pipe.predict_proba(X_fold_val)[:, 1]
        m = compute_all_metrics(y_fold_val.values, probs)

        fold_aucs.append(m["roc_auc"])
        fold_pr_aucs.append(m["pr_auc"])
        fold_f1s.append(m["f1"])

    return {
        "mean_roc_auc": round(float(np.mean(fold_aucs)), 4),
        "std_roc_auc": round(float(np.std(fold_aucs)), 4),
        "mean_pr_auc": round(float(np.mean(fold_pr_aucs)), 4),
        "mean_f1": round(float(np.mean(fold_f1s)), 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs]
    }


def run_all_experiments(csv_path: str = "data/raw/transactions.csv", seed: int = 42) -> Dict[str, Any]:
    """
    Executes formal progressive ML experiment suite.
    """
    print("\n" + "=" * 70)
    print(" 🧪 RECOVERAI ML EXPERIMENTATION & MODEL EVOLUTION PIPELINE")
    print("=" * 70)

    # 1. Load data
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Generating synthetic dataset...")
        df_raw = generate_pipeline(rows=20000, seed=seed)
    else:
        print(f"Loading raw dataset from: {csv_path}")
        df_raw = pd.read_csv(csv_path)

    # Filter failed payments
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    total_failed_count = len(failed_df)

    # Enrich dataframe with non-leaking engineered features
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    # Create experiments artifact directory
    exp_dir = os.path.join("ml", "models", "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    # Stratified 85 / 15 Train-Development vs Untouched Test Split
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=seed, stratify=y_all
    )

    # Within Development set (3,323 samples), use 82.35% Train / 17.65% Val (70% Train, 15% Val overall)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_dev, y_dev, test_size=0.17647, random_state=seed, stratify=y_dev
    )

    print(f"\n📊 Dataset Split (Seed = {seed}):")
    print(f"   - Dev Set   : {len(idx_dev):,} samples (85%) [5-Fold Cross-Validation]")
    print(f"     • Train   : {len(idx_train):,} samples (70%)")
    print(f"     • Val     : {len(idx_val):,} samples (15%)")
    print(f"   - Test Set  : {len(idx_test):,} samples (15%) [UNTOUCHED FOR FINAL EVALUATION]")

    experiments_log: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    # -------------------------------------------------------------------------
    # EXPERIMENT 0: SIMPLE BASELINE (Logistic Regression, 5 Basic Features)
    # -------------------------------------------------------------------------
    exp0_num = ["amount", "hour", "day_of_week"]
    exp0_cat = ["payment_method", "failure_reason"]
    exp0_features = exp0_num + exp0_cat

    audit_features(exp0_features)
    X0 = failed_df[exp0_features]

    X0_dev, y0_dev = X0.loc[idx_dev], y_dev
    X0_train, y0_train = X0.loc[idx_train], y_train
    X0_val, y0_val = X0.loc[idx_val], y_val

    cv0 = run_cross_validation(
        X0_dev, y0_dev, exp0_num, exp0_cat,
        model_factory=lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        seed=seed
    )

    clf0 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    prep0 = build_custom_preprocessor(exp0_num, exp0_cat)
    pipe0 = Pipeline([("preprocessor", prep0), ("classifier", clf0)])

    pipe0.fit(X0_dev, y0_dev)
    joblib.dump(pipe0, os.path.join(exp_dir, "exp_0_baseline.joblib"))
    trained_pipelines["EXP_0"] = pipe0

    exp0_record = {
        "experiment_id": "EXP_0",
        "name": "baseline_logistic_regression",
        "model": "Logistic Regression",
        "feature_count": len(exp0_features),
        "features": exp0_features,
        "changes": "Simple benchmark using 5 basic transaction features.",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t0_fit, 4),
        "metrics": m0,
        "cross_validation": cv0
    }
    experiments_log.append(exp0_record)
    print(f"\n✅ EXP 0 Complete: Baseline Logistic Regression (CV ROC-AUC: {cv0['mean_roc_auc']} ± {cv0['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: FULL FEATURE BENCHMARK (Random Forest, 29 Raw Features)
    # -------------------------------------------------------------------------
    exp1_features = APPROVED_MODEL_FEATURES
    audit_features(exp1_features)
    X1 = failed_df[exp1_features]

    X1_dev, y1_dev = X1.loc[idx_dev], y_dev
    X1_train, y1_train = X1.loc[idx_train], y_train
    X1_val, y1_val = X1.loc[idx_val], y_val

    cv1 = run_cross_validation(
        X1_dev, y1_dev, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        model_factory=lambda: RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1),
        seed=seed
    )

    clf1 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    prep1 = build_custom_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipe1 = Pipeline([("preprocessor", prep1), ("classifier", clf1)])

    t1 = time.time()
    pipe1.fit(X1_train, y1_train)
    t1_fit = time.time() - t1

    val_probs1 = pipe1.predict_proba(X1_val)[:, 1]
    m1 = compute_all_metrics(y1_val.values, val_probs1)

    joblib.dump(pipe1, os.path.join(exp_dir, "exp_1_random_forest.joblib"))
    trained_pipelines["EXP_1"] = pipe1

    exp1_record = {
        "experiment_id": "EXP_1",
        "name": "full_feature_random_forest",
        "model": "Random Forest",
        "feature_count": len(exp1_features),
        "features": exp1_features,
        "changes": "Full 29 approved raw features without hyperparameter tuning.",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t1_fit, 4),
        "metrics": m1,
        "cross_validation": cv1
    }
    experiments_log.append(exp1_record)
    print(f"✅ EXP 1 Complete: Full Feature Random Forest (CV ROC-AUC: {cv1['mean_roc_auc']} ± {cv1['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: HIGH-SIGNAL FEATURE SELECTION (Random Forest, 13 Core Features)
    # -------------------------------------------------------------------------
    exp2_features = HIGH_SIGNAL_MODEL_FEATURES
    audit_features(exp2_features)
    X2 = failed_df[exp2_features]

    X2_dev, y2_dev = X2.loc[idx_dev], y_dev
    X2_train, y2_train = X2.loc[idx_train], y_train
    X2_val, y2_val = X2.loc[idx_val], y_val

    cv2 = run_cross_validation(
        X2_dev, y2_dev, HIGH_SIGNAL_NUMERIC_FEATURES, HIGH_SIGNAL_CATEGORICAL_FEATURES,
        model_factory=lambda: RandomForestClassifier(n_estimators=150, max_depth=10, random_state=seed, n_jobs=-1),
        seed=seed
    )

    clf2 = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=seed, n_jobs=-1)
    prep2 = build_custom_preprocessor(HIGH_SIGNAL_NUMERIC_FEATURES, HIGH_SIGNAL_CATEGORICAL_FEATURES)
    pipe2 = Pipeline([("preprocessor", prep2), ("classifier", clf2)])

    t2 = time.time()
    pipe2.fit(X2_train, y2_train)
    t2_fit = time.time() - t2

    val_probs2 = pipe2.predict_proba(X2_val)[:, 1]
    m2 = compute_all_metrics(y2_val.values, val_probs2)

    joblib.dump(pipe2, os.path.join(exp_dir, "exp_2_high_signal_selection.joblib"))
    trained_pipelines["EXP_2"] = pipe2

    exp2_record = {
        "experiment_id": "EXP_2",
        "name": "high_signal_feature_selection",
        "model": "Random Forest (Feature Selection)",
        "feature_count": len(exp2_features),
        "features": exp2_features,
        "changes": "Purged 15 uninformative noise features. Kept 13 core signal features.",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t2_fit, 4),
        "metrics": m2,
        "cross_validation": cv2
    }
    experiments_log.append(exp2_record)
    print(f"✅ EXP 2 Complete: High-Signal Feature Selection (CV ROC-AUC: {cv2['mean_roc_auc']} ± {cv2['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: ADVANCED ENGINEERED FEATURES (Random Forest, 34 Features)
    # -------------------------------------------------------------------------
    exp3_features = ENGINEERED_MODEL_FEATURES
    audit_features(exp3_features)
    X3 = failed_df[exp3_features]

    X3_dev, y3_dev = X3.loc[idx_dev], y_dev
    X3_train, y3_train = X3.loc[idx_train], y_train
    X3_val, y3_val = X3.loc[idx_val], y_val

    cv3 = run_cross_validation(
        X3_dev, y3_dev, ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        model_factory=lambda: RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1),
        seed=seed
    )

    clf3 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
    prep3 = build_custom_preprocessor(ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipe3 = Pipeline([("preprocessor", prep3), ("classifier", clf3)])

    t3 = time.time()
    pipe3.fit(X3_train, y3_train)
    t3_fit = time.time() - t3

    val_probs3 = pipe3.predict_proba(X3_val)[:, 1]
    m3 = compute_all_metrics(y3_val.values, val_probs3)

    joblib.dump(pipe3, os.path.join(exp_dir, "exp_3_engineered_features.joblib"))
    trained_pipelines["EXP_3"] = pipe3

    exp3_record = {
        "experiment_id": "EXP_3",
        "name": "engineered_feature_model",
        "model": "Random Forest (Advanced Engineered)",
        "feature_count": len(exp3_features),
        "features": exp3_features,
        "changes": "Added 8 non-leaking engineered features (amount_log, streak, risk_velocity_product, high_risk_flag, etc.).",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t3_fit, 4),
        "metrics": m3,
        "cross_validation": cv3
    }
    experiments_log.append(exp3_record)
    print(f"✅ EXP 3 Complete: Advanced Engineered Features (CV ROC-AUC: {cv3['mean_roc_auc']} ± {cv3['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: TUNED GRADIENT BOOSTING CANDIDATE (Tuned XGBoost, 34 Features)
    # -------------------------------------------------------------------------
    exp4_features = ENGINEERED_MODEL_FEATURES
    audit_features(exp4_features)
    X4 = failed_df[exp4_features]

    X4_dev, y4_dev = X4.loc[idx_dev], y_dev
    X4_train, y4_train = X4.loc[idx_train], y_train
    X4_val, y4_val = X4.loc[idx_val], y_val

    prep4 = build_custom_preprocessor(ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X4_train_trans = prep4.fit_transform(X4_train)

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    base_xgb = XGBClassifier(random_state=seed, eval_metric="logloss")
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_grid,
        n_iter=6,
        scoring="roc_auc",
        cv=3,
        random_state=seed,
        n_jobs=-1
    )

    t4 = time.time()
    search.fit(X4_train_trans, y4_train)
    t4_fit = time.time() - t4

    best_xgb = search.best_estimator_
    pipe4 = Pipeline([("preprocessor", prep4), ("classifier", best_xgb)])

    cv4 = run_cross_validation(
        X4_dev, y4_dev, ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        model_factory=lambda: XGBClassifier(**best_xgb.get_params()),
        seed=seed
    )

    val_probs4 = pipe4.predict_proba(X4_val)[:, 1]
    m4 = compute_all_metrics(y4_val.values, val_probs4)

    joblib.dump(pipe4, os.path.join(exp_dir, "exp_4_tuned_xgboost.joblib"))
    trained_pipelines["EXP_4"] = pipe4

    exp4_record = {
        "experiment_id": "EXP_4",
        "name": "tuned_xgboost",
        "model": "Tuned XGBoost",
        "feature_count": len(exp4_features),
        "features": exp4_features,
        "best_hyperparameters": search.best_params_,
        "changes": "Tuned via RandomizedSearchCV on train fold with CV validation.",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t4_fit, 4),
        "metrics": m4,
        "cross_validation": cv4
    }
    experiments_log.append(exp4_record)
    print(f"✅ EXP 4 Complete: Tuned XGBoost (CV ROC-AUC: {cv4['mean_roc_auc']} ± {cv4['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: CALIBRATED CLASSIFIER CANDIDATE (Platt / Isotonic Calibration)
    # -------------------------------------------------------------------------
    exp5_features = ENGINEERED_MODEL_FEATURES
    audit_features(exp5_features)
    X5 = failed_df[exp5_features]

    X5_dev, y5_dev = X5.loc[idx_dev], y_dev
    X5_train, y5_train = X5.loc[idx_train], y_train
    X5_val, y5_val = X5.loc[idx_val], y_val

    prep5 = build_custom_preprocessor(ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X5_train_trans = prep5.fit_transform(X5_train)
    X5_val_trans = prep5.transform(X5_val)

    calib_clf = CalibratedClassifierCV(estimator=best_xgb, method="sigmoid", cv=3)
    t5 = time.time()
    calib_clf.fit(X5_train_trans, y5_train)
    t5_fit = time.time() - t5

    pipe5 = Pipeline([("preprocessor", prep5), ("classifier", calib_clf)])

    cv5 = run_cross_validation(
        X5_dev, y5_dev, ENGINEERED_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        model_factory=lambda: CalibratedClassifierCV(estimator=best_xgb, method="sigmoid", cv=3),
        seed=seed
    )

    val_probs5 = pipe5.predict_proba(X5_val)[:, 1]
    m5 = compute_all_metrics(y5_val.values, val_probs5)

    joblib.dump(pipe5, os.path.join(exp_dir, "exp_5_calibrated_model.joblib"))
    trained_pipelines["EXP_5"] = pipe5

    exp5_record = {
        "experiment_id": "EXP_5",
        "name": "calibrated_model",
        "model": "Calibrated XGBoost (Platt Scaling)",
        "feature_count": len(exp5_features),
        "features": exp5_features,
        "changes": "Sigmoidal probability calibration applied to tuned XGBoost candidate.",
        "selection_dataset": "cross_validation",
        "fit_time_seconds": round(t5_fit, 4),
        "metrics": m5,
        "cross_validation": cv5
    }
    experiments_log.append(exp5_record)
    print(f"✅ EXP 5 Complete: Calibrated Model (CV ROC-AUC: {cv5['mean_roc_auc']} ± {cv5['std_roc_auc']})")

    # -------------------------------------------------------------------------
    # PERSIST EXPERIMENT HISTORY JSON
    # -------------------------------------------------------------------------
    history_payload = {
        "dataset": {
            "name": "synthetic_transactions_20k",
            "synthetic": True,
            "seed": seed,
            "total_rows": len(df_raw),
            "failed_payment_rows": total_failed_count
        },
        "split": {
            "development": len(idx_dev),
            "train": len(idx_train),
            "validation": len(idx_val),
            "test": len(idx_test)
        },
        "experiments": experiments_log
    }

    history_path = os.path.join("evaluation", "experiment_history.json")
    os.makedirs("evaluation", exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history_payload, f, indent=2)
    print(f"\n💾 Saved experiment history to: {history_path}")

    # -------------------------------------------------------------------------
    # FINAL MODEL SELECTION & UNTOUCHED TEST EVALUATION
    # -------------------------------------------------------------------------
    best_exp = max(experiments_log, key=lambda e: (e["cross_validation"]["mean_roc_auc"], e["cross_validation"]["mean_pr_auc"]))
    best_exp_id = best_exp["experiment_id"]
    best_pipeline = trained_pipelines[best_exp_id]

    print(f"\n🏆 Selected Winner Candidate: {best_exp_id} ({best_exp['name']}) based on 5-Fold CV ROC-AUC {best_exp['cross_validation']['mean_roc_auc']} ± {best_exp['cross_validation']['std_roc_auc']}")
    print(" 🎯 Evaluating Baseline vs Selected Candidate ONCE on untouched Test set...")

    # Evaluate Baseline (EXP_0) on Test Set
    X0_test = failed_df.loc[idx_test, exp0_features]
    test_probs0 = pipe0.predict_proba(X0_test)[:, 1]
    baseline_test_m = compute_all_metrics(y_test.values, test_probs0)

    # Evaluate Selected Candidate on Test Set
    if best_exp_id == "EXP_0":
        best_X_test = X0_test
    elif best_exp_id == "EXP_1":
        best_X_test = failed_df.loc[idx_test, exp1_features]
    elif best_exp_id == "EXP_2":
        best_X_test = failed_df.loc[idx_test, exp2_features]
    else:
        best_X_test = failed_df.loc[idx_test, exp3_features]

    test_probs_best = best_pipeline.predict_proba(best_X_test)[:, 1]
    final_test_m = compute_all_metrics(y_test.values, test_probs_best)

    roc_abs_diff = final_test_m["roc_auc"] - baseline_test_m["roc_auc"]
    roc_pct_diff = (roc_abs_diff / baseline_test_m["roc_auc"] * 100) if baseline_test_m["roc_auc"] > 0 else 0.0

    f1_abs_diff = final_test_m["f1"] - baseline_test_m["f1"]
    f1_pct_diff = (f1_abs_diff / baseline_test_m["f1"] * 100) if baseline_test_m["f1"] > 0 else 0.0

    # Top Feature Importance for tree model
    top_features = []
    try:
        classifier = best_pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            preprocessor = best_pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = classifier.feature_importances_
            sorted_indices = np.argsort(importances)[::-1][:10]
            top_features = [[str(feature_names[i]), round(float(importances[i]), 4)] for i in sorted_indices]
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

    final_comparison_payload = {
        "selected_candidate_experiment_id": best_exp_id,
        "selected_candidate_name": best_exp["name"],
        "baseline_test_metrics": baseline_test_m,
        "final_model_test_metrics": final_test_m,
        "improvement": {
            "roc_auc_absolute": round(roc_abs_diff, 4),
            "roc_auc_percent": round(roc_pct_diff, 2),
            "f1_absolute": round(f1_abs_diff, 4),
            "f1_percent": round(f1_pct_diff, 2)
        },
        "top_10_features": top_features
    }

    final_comp_path = os.path.join("evaluation", "final_model_comparison.json")
    with open(final_comp_path, "w") as f:
        json.dump(final_comparison_payload, f, indent=2)
    print(f"💾 Saved final model comparison to: {final_comp_path}")

    return history_payload


if __name__ == "__main__":
    run_all_experiments()
