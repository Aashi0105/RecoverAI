"""
Main ML Training, Model Selection, and Business Evaluation Pipeline for RecoverAI.

Runs end-to-end:
1. Loads raw payment data (filters failed payments only).
2. Audits feature schema for target leakage.
3. Performs 70/15/15 stratified split.
4. Trains & compares Logistic Regression, Random Forest, and XGBoost models.
5. Selects best candidate based on Validation ROC-AUC and Calibration.
6. Evaluates selected model ONCE on the untouched Test set.
7. Computes business ROI metrics & decision threshold trade-offs.
8. Extracts feature importances and saves diagnostic plots.
9. Serializes final pipeline artifact to ml/models/recovery_model.joblib.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml.features import prepare_features_and_target, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, APPROVED_MODEL_FEATURES
from evaluation.model_metrics import compute_all_metrics, save_evaluation_plots
from evaluation.business_metrics import calculate_business_impact, evaluate_decision_thresholds
from data.generate_data import generate_pipeline


def train_and_evaluate_ml_pipeline(csv_path: str = "data/raw/transactions.csv", seed: int = 42):
    """
    Main training execution pipeline.
    """
    print("\n" + "=" * 70)
    print(" 🤖 RECOVERAI MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Ensure dataset exists
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Generating synthetic dataset...")
        df_raw = generate_pipeline(rows=20000, seed=seed)
    else:
        print(f"Loading raw dataset from: {csv_path}")
        df_raw = pd.read_csv(csv_path)

    # 2. Filter failed payments & extract audited features
    X, y = prepare_features_and_target(df_raw)

    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    amounts_all = failed_df["amount"].values

    total_failed_count = len(failed_df)
    print(f"\n📌 Failed payments extracted for ML model: {total_failed_count:,}")
    print(f"📌 Target distribution ('recovered'):")
    rec_count = int(y.sum())
    not_rec_count = total_failed_count - rec_count
    print(f"   - Recovered (1)    : {rec_count:,} ({rec_count/total_failed_count*100:.2f}%)")
    print(f"   - Not Recovered (0): {not_rec_count:,} ({not_rec_count/total_failed_count*100:.2f}%)")

    # 3. Train/Val/Test Stratified Split (70 / 15 / 15)
    # First split 70% train, 30% temp
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, failed_df.index, test_size=0.30, random_state=seed, stratify=y
    )
    # Split temp 50/50 into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    amounts_test = failed_df.loc[idx_test, "amount"].values

    print(f"\n📊 Dataset Split:")
    print(f"   - Training Set   : {len(X_train):,} samples (70%)")
    print(f"   - Validation Set : {len(X_val):,} samples (15%)")
    print(f"   - Test Set       : {len(X_test):,} samples (15%)")

    # 4. Define Candidate Models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=seed, eval_metric="logloss")
    }

    val_results = {}
    fitted_pipelines = {}

    print("\n" + "-" * 70)
    print(" 🛠️ TRAINING CANDIDATE MODELS & EVALUATING ON VALIDATION SET")
    print("-" * 70)

    for name, clf in candidates.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        pipeline.fit(X_train, y_train)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        
        metrics = compute_all_metrics(y_val.values, val_probs)
        val_results[name] = metrics
        fitted_pipelines[name] = pipeline

    # Print Candidate Comparison Table
    print("\n📋 Candidate Model Validation Comparison Table:")
    print(f"{'Model':<22} | {'Precision':<10} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
    print("-" * 84)
    for name, m in val_results.items():
        print(f"{name:<22} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['brier_score']:<8.4f}")
    print("-" * 84)

    # 5. Select Best Model (based on Validation ROC-AUC & Calibration)
    best_name = max(val_results, key=lambda k: (val_results[k]["roc_auc"], val_results[k]["pr_auc"]))
    best_pipeline = fitted_pipelines[best_name]
    best_val_metrics = val_results[best_name]

    print(f"\n🏆 Selected Model: {best_name}")
    print(f"   Reason: Highest combined ROC-AUC ({best_val_metrics['roc_auc']:.4f}) and PR-AUC ({best_val_metrics['pr_auc']:.4f}) on Validation set.")

    # 6. Evaluate Selected Model ONCE on the untouched Test Set
    print("\n" + "=" * 70)
    print( " 🧪 FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_all_metrics(y_test.values, test_probs)

    print(f" Test Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f" Test Precision  : {test_metrics['precision']:.4f}")
    print(f" Test Recall     : {test_metrics['recall']:.4f}")
    print(f" Test F1 Score   : {test_metrics['f1']:.4f}")
    print(f" Test ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f" Test PR-AUC     : {test_metrics['pr_auc']:.4f}")
    print(f" Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f" False Pos Rate  : {test_metrics['fpr']:.4f}")
    print(f" False Neg Rate  : {test_metrics['fnr']:.4f}")

    # 7. Extract Feature Importances (Top 10)
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    # Retrieve transformed feature names
    cat_onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_onehot.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    feature_importances_dict = {}
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importances_dict = dict(zip(all_feature_names, importances))
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        feature_importances_dict = dict(zip(all_feature_names, importances))

    sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n🔝 Top 10 Most Important Features:")
    for rank, (feat, score) in enumerate(sorted_features[:10], 1):
        print(f"  {rank:2d}. {feat:<40}: {score:.4f}")

    # 8. Business Financial Metrics & Threshold Trade-offs
    print("\n" + "=" * 70)
    print(" 💰 BUSINESS FINANCIAL IMPACT & THRESHOLD ANALYSIS (TEST SET)")
    print("=" * 70)

    biz_impact = calculate_business_impact(amounts_test, y_test.values, test_probs, default_threshold=0.50)
    print(f" Failed payments evaluated         : {biz_impact['total_failed_count']:,}")
    print(f" Failed transaction value          : ₹{biz_impact['total_failed_value']:,.2f}")
    print(f" Actual recovered revenue          : ₹{biz_impact['actual_recovered_revenue']:,.2f}")
    print(f" Predicted recoverable revenue     : ₹{biz_impact['predicted_recoverable_revenue']:,.2f}")
    print(f" Expected recovery value (sum P*A) : ₹{biz_impact['expected_recovery_value']:,.2f}")
    print(f" Revenue capture recall             : {biz_impact['revenue_capture_recall']*100:.2f}%")
    print(f" Revenue precision                  : {biz_impact['revenue_precision']*100:.2f}%")

    threshold_results = evaluate_decision_thresholds(amounts_test, y_test.values, test_probs)

    print("\n📊 Decision Threshold Trade-off Table:")
    print(f"{'Threshold':<10} | {'Selected Txns':<15} | {'% Selected':<12} | {'Revenue Selected':<18} | {'Actual Recovered':<18} | {'Recovery Rate':<14} | {'Missed Revenue':<16}")
    print("-" * 115)
    for tr in threshold_results:
        print(f"{tr['threshold']:<10.2f} | {tr['payments_selected']:<15,d} | {tr['percentage_selected']:<12.2f}% | ₹{tr['revenue_selected']:<17,.2f} | ₹{tr['actual_recovered_revenue']:<17,.2f} | {tr['recovery_rate_pct']:<14.2f}% | ₹{tr['missed_recoverable_revenue']:<15,.2f}")
    print("-" * 115)

    # 9. Save Diagnostic Plots
    plot_paths = save_evaluation_plots(y_test.values, test_probs, feature_importances_dict)
    print("\n📈 Saved Evaluation Diagnostic Plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name:<20}: {plot_path}")

    # 10. Save Final Model Artifact & Metadata
    model_dir = "ml/models"
    os.makedirs(model_dir, exist_ok=True)
    artifact_path = os.path.join(model_dir, "recovery_model.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    joblib.dump({
        "pipeline": best_pipeline,
        "selected_model": best_name,
        "feature_names": APPROVED_MODEL_FEATURES,
        "metrics": test_metrics
    }, artifact_path)

    metadata = {
        "selected_model": best_name,
        "validation_metrics": val_results,
        "test_metrics": test_metrics,
        "top_10_features": sorted_features[:10],
        "business_impact": biz_impact,
        "threshold_analysis": threshold_results
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Model artifact saved to : {artifact_path}")
    print(f"📄 Metadata saved to       : {metadata_path}")
    print("=" * 70 + "\n")

    return best_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate_ml_pipeline()
