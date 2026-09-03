"""
Regression tests for verifying the feature schema contracts of both model artifacts:
1. ml/models/recovery_model.joblib
   Must have a preprocessing/model input schema exactly matching APPROVED_MODEL_FEATURES.
2. ml/models/experiments/exp_0_baseline.joblib
   Must have a preprocessing schema exactly matching EXP0_NUMERIC_FEATURES + EXP0_CATEGORICAL_FEATURES.

Strictly read-only: does not modify, retrain, or overwrite model artifacts or datasets.
"""

import os
import sys
import pytest
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import (
    APPROVED_MODEL_FEATURES,
    EXP0_NUMERIC_FEATURES,
    EXP0_CATEGORICAL_FEATURES,
)


def test_recovery_model_artifact_schema_contract():
    """
    Verifies that ml/models/recovery_model.joblib exists and its fitted preprocessor
    pipeline explicitly expects input features exactly matching APPROVED_MODEL_FEATURES.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml", "models", "recovery_model.joblib")
    assert os.path.exists(model_path), f"Artifact missing: {model_path}"

    artifact = joblib.load(model_path)
    assert isinstance(artifact, dict), "recovery_model.joblib expected to be a dictionary artifact"
    assert "pipeline" in artifact, "Artifact missing 'pipeline' key"

    pipeline = artifact["pipeline"]
    assert "preprocessor" in pipeline.named_steps, "Pipeline missing 'preprocessor' step"

    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = dict((name, cols) for name, trans, cols in preprocessor.transformers)

    assert "num" in transformers, "Preprocessor missing 'num' transformer"
    assert "cat" in transformers, "Preprocessor missing 'cat' transformer"

    num_cols = list(transformers["num"])
    cat_cols = list(transformers["cat"])
    actual_preprocessor_features = num_cols + cat_cols

    # Verify exact feature count, exact names, and exact order
    assert actual_preprocessor_features == APPROVED_MODEL_FEATURES, (
        f"Mismatch between recovery_model preprocessor schema and APPROVED_MODEL_FEATURES.\n"
        f"Actual: {actual_preprocessor_features}\n"
        f"Expected: {APPROVED_MODEL_FEATURES}"
    )

    # If artifact stores feature_names metadata, verify it matches as well
    if "feature_names" in artifact:
        assert artifact["feature_names"] == APPROVED_MODEL_FEATURES, (
            f"Mismatch between artifact['feature_names'] metadata and APPROVED_MODEL_FEATURES.\n"
            f"Actual: {artifact['feature_names']}\n"
            f"Expected: {APPROVED_MODEL_FEATURES}"
        )

    # Verify model forward pass with exact schema
    dummy_input = {col: [1.0 if col in num_cols else "upi"] for col in APPROVED_MODEL_FEATURES}
    df_dummy = pd.DataFrame(dummy_input)
    prob = pipeline.predict_proba(df_dummy)[0, 1]
    assert 0.0 <= prob <= 1.0, f"Predicted probability {prob} out of bounds [0, 1]"


def test_exp0_baseline_artifact_schema_contract():
    """
    Verifies that ml/models/experiments/exp_0_baseline.joblib exists and its fitted preprocessor
    pipeline explicitly expects input features exactly matching EXP0_NUMERIC_FEATURES + EXP0_CATEGORICAL_FEATURES.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib")
    assert os.path.exists(model_path), f"Artifact missing: {model_path}"

    pipeline = joblib.load(model_path)
    assert hasattr(pipeline, "named_steps"), "exp_0_baseline.joblib expected to be a Pipeline"
    assert "preprocessor" in pipeline.named_steps, "Pipeline missing 'preprocessor' step"

    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = dict((name, cols) for name, trans, cols in preprocessor.transformers)

    assert "num" in transformers, "Preprocessor missing 'num' transformer"
    assert "cat" in transformers, "Preprocessor missing 'cat' transformer"

    num_cols = list(transformers["num"])
    cat_cols = list(transformers["cat"])
    actual_features = num_cols + cat_cols
    expected_features = EXP0_NUMERIC_FEATURES + EXP0_CATEGORICAL_FEATURES

    # Verify numeric features exact match and order
    assert num_cols == EXP0_NUMERIC_FEATURES, (
        f"Mismatch in EXP0 numeric features.\nActual: {num_cols}\nExpected: {EXP0_NUMERIC_FEATURES}"
    )

    # Verify categorical features exact match and order
    assert cat_cols == EXP0_CATEGORICAL_FEATURES, (
        f"Mismatch in EXP0 categorical features.\nActual: {cat_cols}\nExpected: {EXP0_CATEGORICAL_FEATURES}"
    )

    # Verify combined features exact match and order
    assert actual_features == expected_features, (
        f"Mismatch between exp_0_baseline preprocessor schema and EXP0 features.\n"
        f"Actual: {actual_features}\n"
        f"Expected: {expected_features}"
    )

    # Verify model forward pass with exact schema
    dummy_input = {
        "amount": [1000.0],
        "hour": [12],
        "day_of_week": [2],
        "payment_method": ["upi"],
        "failure_reason": ["network_timeout"]
    }
    df_dummy = pd.DataFrame(dummy_input)
    prob = pipeline.predict_proba(df_dummy)[0, 1]
    assert 0.0 <= prob <= 1.0, f"Predicted probability {prob} out of bounds [0, 1]"


def test_model_artifacts_scikit_learn_version_consistency():
    """
    Strict reproducibility test ensuring both serialized model artifacts load under
    the pinned scikit-learn runtime without raising an InconsistentVersionWarning.
    """
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning

    artifacts_to_verify = [
        os.path.join(PROJECT_ROOT, "ml", "models", "recovery_model.joblib"),
        os.path.join(PROJECT_ROOT, "ml", "models", "experiments", "exp_0_baseline.joblib"),
    ]

    for model_path in artifacts_to_verify:
        assert os.path.exists(model_path), f"Artifact missing: {model_path}"
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            loaded = joblib.load(model_path)
            assert loaded is not None

            version_warnings = [
                w for w in recorded_warnings if issubclass(w.category, InconsistentVersionWarning)
            ]
            assert not version_warnings, (
                f"Artifact {os.path.basename(model_path)} emitted InconsistentVersionWarning: "
                f"{[str(w.message) for w in version_warnings]}"
            )
