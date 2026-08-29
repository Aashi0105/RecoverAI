import os
import pytest
import numpy as np
import pandas as pd
import joblib

from data.generate_data import generate_pipeline
from ml.features import (
    prepare_features_and_target,
    audit_features,
    APPROVED_MODEL_FEATURES,
    FORBIDDEN_TARGET_LEAKAGE_COLUMNS
)
from ml.train import train_and_evaluate_ml_pipeline
from ml.predict import predict_recovery_probability, load_recovery_model


@pytest.fixture(scope="module")
def setup_dataset_and_model(tmp_path_factory):
    """Fixture to generate synthetic data and train a lightweight model artifact."""
    data_dir = tmp_path_factory.mktemp("data")
    csv_path = os.path.join(data_dir, "transactions.csv")
    
    # Generate 500 rows for fast testing
    df_raw = generate_pipeline(rows=500, seed=42, output_dir=str(data_dir))
    pipeline, metadata = train_and_evaluate_ml_pipeline(csv_path=csv_path, seed=42)
    return csv_path, df_raw, pipeline, metadata


def test_dataset_loads_and_filters_failed_payments(setup_dataset_and_model):
    """1 & 2: Test dataset loads and filters to failed payments only."""
    csv_path, df_raw, _, _ = setup_dataset_and_model
    X, y = prepare_features_and_target(df_raw)
    
    failed_count = (df_raw["payment_status"] == "failed").sum()
    assert len(X) == failed_count
    assert len(y) == failed_count


def test_target_exists(setup_dataset_and_model):
    """3: Test target 'recovered' exists and is binary."""
    _, df_raw, _, _ = setup_dataset_and_model
    X, y = prepare_features_and_target(df_raw)
    assert set(y.unique()).issubset({0, 1})


def test_forbidden_target_leakage_audit():
    """4: Test forbidden target-leakage audit raises error when target fields are present."""
    bad_features = APPROVED_MODEL_FEATURES + ["recovered"]
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        audit_features(bad_features)
        
    bad_features_2 = APPROVED_MODEL_FEATURES + ["recovery_action"]
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        audit_features(bad_features_2)


def test_feature_transformation(setup_dataset_and_model):
    """5: Test feature extraction and ColumnTransformer pipeline transformation."""
    _, df_raw, pipeline, _ = setup_dataset_and_model
    X, y = prepare_features_and_target(df_raw)
    
    # Test preprocessor output shape
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] > 0


def test_model_training_and_probability_bounds(setup_dataset_and_model):
    """6 & 7: Test candidate models train and predict probabilities between 0 and 1."""
    _, df_raw, pipeline, _ = setup_dataset_and_model
    X, y = prepare_features_and_target(df_raw)
    
    probs = pipeline.predict_proba(X)[:, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_predict_recovery_probability_single_transaction(setup_dataset_and_model):
    """8: Test probability prediction works on a single transaction dictionary."""
    _, df_raw, _, _ = setup_dataset_and_model
    failed_row = df_raw[df_raw["payment_status"] == "failed"].iloc[0].to_dict()
    
    # Remove leakage fields from dict
    for col in FORBIDDEN_TARGET_LEAKAGE_COLUMNS:
        failed_row.pop(col, None)
        
    res = predict_recovery_probability(failed_row)
    assert "recovery_probability" in res
    prob = res["recovery_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_model_artifact_save_and_reload(setup_dataset_and_model):
    """9 & 10: Test model artifact saves, reloads, and gives identical predictions."""
    _, df_raw, pipeline, _ = setup_dataset_and_model
    failed_row = df_raw[df_raw["payment_status"] == "failed"].iloc[0].to_dict()
    for col in FORBIDDEN_TARGET_LEAKAGE_COLUMNS:
        failed_row.pop(col, None)

    # Predict using original loaded pipeline
    df_single = pd.DataFrame([failed_row])[APPROVED_MODEL_FEATURES]
    orig_prob = pipeline.predict_proba(df_single)[0, 1]

    # Reload from disk artifact
    artifact = load_recovery_model()
    reloaded_pipeline = artifact["pipeline"]
    reloaded_prob = reloaded_pipeline.predict_proba(df_single)[0, 1]

    assert np.isclose(orig_prob, reloaded_prob, atol=1e-6)
