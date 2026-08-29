"""
Evaluation Metrics & Visualization Utilities for ML Payment Recovery Model.

Computes classification metrics, probability calibration scores, and renders
diagnostic plots (ROC, PR, Confusion Matrix, Calibration, Feature Importance).
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


def compute_all_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics, error rates, and probability calibration.
    """
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    
    cm = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4)
    }


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    feature_importances: Optional[Dict[str, float]] = None,
    output_dir: str = "evaluation/results"
) -> Dict[str, str]:
    """
    Renders and saves evaluation diagnostic plots:
    1. ROC Curve
    2. Precision-Recall Curve
    3. Confusion Matrix
    4. Calibration Curve
    5. Top 10 Feature Importances (if provided)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_vals, tpr_vals, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    saved_paths["roc_curve"] = roc_path

    # 2. Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_vals, prec_vals, color="#2ca02c", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "pr_curve.png")
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)
    saved_paths["pr_curve"] = pr_path

    # 3. Confusion Matrix
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Recovered (0)", "Recovered (1)"])
    ax.set_yticklabels(["Not Recovered (0)", "Recovered (1)"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Threshold = 0.50)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    saved_paths["confusion_matrix"] = cm_path

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#ff7f0e", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration Curve")
    ax.legend(loc="upper left")
    plt.tight_layout()
    calib_path = os.path.join(output_dir, "calibration_curve.png")
    fig.savefig(calib_path, dpi=200)
    plt.close(fig)
    saved_paths["calibration_curve"] = calib_path

    # 5. Feature Importance Plot (if available)
    if feature_importances:
        top_feats = pd.Series(feature_importances).sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        top_feats.plot(kind="barh", ax=ax, color="#9467bd")
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Most Important Recovery Features")
        plt.tight_layout()
        fi_path = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(fi_path, dpi=200)
        plt.close(fig)
        saved_paths["feature_importance"] = fi_path

    return saved_paths
