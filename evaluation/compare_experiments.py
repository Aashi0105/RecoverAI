"""
Experiment Comparison & Model Evolution Utility.

Loads evaluation/experiment_history.json and renders structured comparison tables
and candidate ranking reports.
"""

import json
import os
from typing import Dict, Any


def load_experiment_history(path: str = "evaluation/experiment_history.json") -> Dict[str, Any]:
    """Loads experiment history JSON payload."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment history file not found at {path}. Run ml/experiments.py first.")
    with open(path, "r") as f:
        return json.load(f)


def render_experiment_comparison(history: Dict[str, Any] = None) -> None:
    """Renders formatted experiment evolution table and cross-validation rankings."""
    if history is None:
        history = load_experiment_history()

    experiments = history.get("experiments", [])
    if not experiments:
        print("No experiments found in history file.")
        return

    print("\n" + "=" * 90)
    print(" 📊 RECOVERAI MODEL EVOLUTION & 5-FOLD CV COMPARISON TABLE")
    print("=" * 90)

    header = f"{'Exp ID':<7} | {'Name':<28} | {'Feats':<5} | {'5-Fold CV ROC-AUC':<18} | {'Val ROC-AUC':<11} | {'Val F1':<8}"
    print(header)
    print("-" * 90)

    for exp in experiments:
        eid = exp["experiment_id"]
        name = exp["name"]
        fc = exp["feature_count"]
        m = exp["metrics"]
        cv = exp.get("cross_validation", {})
        cv_str = f"{cv.get('mean_roc_auc', 0.0):.4f} ± {cv.get('std_roc_auc', 0.0):.4f}" if cv else "N/A"
        print(f"{eid:<7} | {name:<28} | {fc:<5} | {cv_str:<18} | {m['roc_auc']:<11.4f} | {m['f1']:<8.4f}")

    print("-" * 90)

    print("\n=================================================")
    print(" MODEL EVOLUTION PROGRESSION (5-FOLD CV ROC-AUC)")
    print("=================================================")

    for i, exp in enumerate(experiments):
        eid = exp["experiment_id"]
        model_name = exp["model"]
        fc = exp["feature_count"]
        cv = exp.get("cross_validation", {})
        cv_str = f"{cv.get('mean_roc_auc', 0.0):.4f} ± {cv.get('std_roc_auc', 0.0):.4f}" if cv else f"{exp['metrics']['roc_auc']:.4f}"
        print(f"{eid} : {model_name} ({fc} Features) ──► 5-Fold CV ROC-AUC: {cv_str}")
        if i < len(experiments) - 1:
            print("   ↓")
    print("=================================================\n")


if __name__ == "__main__":
    render_experiment_comparison()
