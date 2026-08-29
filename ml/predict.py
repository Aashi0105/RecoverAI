"""
ML Inference Engine for RecoverAI Payment Recovery System.

Loads serialized model artifact from ml/models/recovery_model.joblib and computes
recovery probability scores for single or batch transaction records.
"""

import os
from typing import Dict, Any, Union
import joblib
import pandas as pd

from ml.features import APPROVED_MODEL_FEATURES, audit_features

# Global cache for loaded model pipeline artifact
_MODEL_ARTIFACT = None
MODEL_PATH = os.path.join("ml", "models", "recovery_model.joblib")


def load_recovery_model():
    """
    Lazy loads and caches the trained model pipeline artifact from disk.
    """
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at '{MODEL_PATH}'. "
                f"Please train the model first by running 'python ml/train.py'."
            )
        _MODEL_ARTIFACT = joblib.load(MODEL_PATH)
    return _MODEL_ARTIFACT


def predict_recovery_probability(transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Computes recovery probability P(recovery_success) for a single or batch failed payment record.

    Input:
      transaction: dict, pd.Series, or pd.DataFrame containing approved features.

    Output:
      {"recovery_probability": float}
    """
    # 1. Convert input dict / Series to DataFrame
    if isinstance(transaction, dict):
        df_input = pd.DataFrame([transaction])
    elif isinstance(transaction, pd.Series):
        df_input = pd.DataFrame([transaction.to_dict()])
    elif isinstance(transaction, pd.DataFrame):
        df_input = transaction.copy()
    else:
        raise TypeError(f"Unsupported transaction input type: {type(transaction)}")

    # 2. Strict Target Leakage Audit
    audit_features(list(df_input.columns))

    # 3. Ensure all approved features exist in input
    missing = [col for col in APPROVED_MODEL_FEATURES if col not in df_input.columns]
    if missing:
        raise ValueError(f"Transaction input missing required approved features: {missing}")

    X_feats = df_input[APPROVED_MODEL_FEATURES]

    # 4. Load pipeline & predict probability
    artifact = load_recovery_model()
    pipeline = artifact["pipeline"]

    prob = float(pipeline.predict_proba(X_feats)[0, 1])
    prob_rounded = round(prob, 4)

    return {
        "recovery_probability": prob_rounded
    }
