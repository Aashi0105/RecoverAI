"""
Unit tests for monitoring/drift_detection.py.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from monitoring.drift_detection import (
    generate_baseline_snapshot,
    calculate_psi,
    run_drift_report,
    SNAPSHOT_PATH
)
from ml.features import add_engineered_features
from sklearn.model_selection import train_test_split


def test_calculate_psi_thresholds():
    """
    Verifies mathematical behavior and threshold classifications for calculate_psi.
    """
    base = [0.25, 0.25, 0.25, 0.25]
    
    # 1. Identical distribution -> PSI near 0.0, STABLE
    psi, status = calculate_psi(base, base)
    assert psi == 0.0
    assert status == "STABLE"

    # 2. Moderate shift -> MODERATE DRIFT
    mod_curr = [0.15, 0.35, 0.15, 0.35]
    psi_mod, status_mod = calculate_psi(base, mod_curr)
    assert 0.10 <= psi_mod < 0.25
    assert status_mod == "MODERATE DRIFT"

    # 3. Severe shift -> SIGNIFICANT DRIFT
    sev_curr = [0.01, 0.90, 0.04, 0.05]
    psi_sev, status_sev = calculate_psi(base, sev_curr)
    assert psi_sev >= 0.25
    assert status_sev == "SIGNIFICANT DRIFT"


def test_baseline_snapshot_generation():
    """
    Verifies baseline snapshot generation on Development set.
    """
    snapshot = generate_baseline_snapshot()
    assert os.path.exists(SNAPSHOT_PATH)
    assert "numeric" in snapshot
    assert "categorical" in snapshot
    assert snapshot["metadata"]["dev_sample_size"] == 3581

    for feat in ["amount", "hour", "day_of_week"]:
        assert feat in snapshot["numeric"]
        assert len(snapshot["numeric"][feat]["percentages"]) > 0

    for feat in ["payment_method", "failure_reason"]:
        assert feat in snapshot["categorical"]
        assert len(snapshot["categorical"][feat]["categories"]) > 0


def test_drift_report_on_untouched_test_set():
    """
    Verifies drift report execution on untouched test set (N=633).
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)
    y_all = failed_df["recovered"].astype(int)

    idx_dev, idx_test, _, _ = train_test_split(
        failed_df.index, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    test_df = failed_df.loc[idx_test].copy()

    report = run_drift_report(test_df)
    assert len(report["features"]) == 5
    assert "system_status" in report
    assert report["significant_drift_count"] == 0  # Test set from same generation process


def test_drift_report_detects_synthetic_shift():
    """
    Verifies that run_drift_report correctly flags significant drift on a synthetic skewed sample.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)

    # Create synthetic skewed sample: filter strictly to high-amount transactions with single failure reason
    skewed_df = failed_df[(failed_df["amount"] > 10000) & (failed_df["failure_reason"] == "insufficient_funds")].copy()

    report = run_drift_report(skewed_df)
    assert report["significant_drift_count"] > 0
    assert "REVIEW RECOMMENDED" in report["system_status"]


def test_drift_recommendation_generation():
    """
    Verifies deterministic recommendation strings for STABLE, MODERATE DRIFT, and SIGNIFICANT DRIFT.
    """
    from monitoring.drift_detection import get_drift_recommendation

    rec_stable = get_drift_recommendation("STABLE", "amount")
    assert "No intervention required" in rec_stable

    rec_mod = get_drift_recommendation("MODERATE DRIFT", "amount")
    assert "Monitor feature 'amount'" in rec_mod
    assert "upstream data pipeline" in rec_mod

    rec_sig = get_drift_recommendation("SIGNIFICANT DRIFT", "amount")
    assert "Investigate upstream data source" in rec_sig
    assert "Schedule model & data distribution review" in rec_sig


def test_drift_event_persistence(tmp_path):
    """
    Verifies that append_drift_audit_event appends valid JSONL records containing required schema fields.
    """
    import json
    from monitoring.drift_detection import append_drift_audit_event

    test_log = os.path.join(tmp_path, "drift_audit.jsonl")
    dummy_summary = {
        "system_status": "HEALTHY",
        "significant_drift_count": 0,
        "moderate_drift_count": 0,
        "features": [
            {
                "feature": "amount",
                "type": "numeric",
                "psi": 0.0215,
                "status": "STABLE",
                "recommendation": "No intervention required."
            }
        ]
    }

    append_drift_audit_event(dummy_summary, log_path=test_log)

    assert os.path.exists(test_log)
    with open(test_log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert "timestamp" in record
    assert record["system_status"] == "HEALTHY"
    assert record["significant_drift_count"] == 0
    assert len(record["monitored_features"]) == 1
    assert record["monitored_features"][0]["feature"] == "amount"
    assert record["monitored_features"][0]["psi"] == 0.0215


def test_drift_event_persistence_append_not_overwrite(tmp_path):
    """
    Verifies that multiple calls to append_drift_audit_event append rather than overwrite.
    """
    import json
    from monitoring.drift_detection import append_drift_audit_event

    test_log = os.path.join(tmp_path, "drift_audit_append.jsonl")
    s1 = {"system_status": "HEALTHY", "significant_drift_count": 0, "features": []}
    s2 = {"system_status": "REVIEW RECOMMENDED", "significant_drift_count": 2, "features": []}

    append_drift_audit_event(s1, log_path=test_log)
    append_drift_audit_event(s2, log_path=test_log)

    with open(test_log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])

    assert r1["system_status"] == "HEALTHY"
    assert r2["system_status"] == "REVIEW RECOMMENDED"


def test_drift_report_persist_true(tmp_path):
    """
    Verifies that run_drift_report(..., persist=True) writes/appends a valid audit event.
    """
    import json
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)

    test_log = os.path.join(tmp_path, "drift_audit_persist_true.jsonl")

    res = run_drift_report(failed_df.iloc[:50], persist=True, log_path=test_log)
    assert "system_status" in res
    assert os.path.exists(test_log)

    with open(test_log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert "timestamp" in event
    assert "system_status" in event
    assert len(event["monitored_features"]) == 5


def test_drift_report_persist_false(tmp_path):
    """
    Verifies that run_drift_report(..., persist=False) computes drift and recommendations
    without writing or modifying any log file.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    df_raw = pd.read_csv(csv_path)
    failed_df = df_raw[df_raw["payment_status"] == "failed"].copy()
    failed_df = add_engineered_features(failed_df)

    test_log = os.path.join(tmp_path, "drift_audit_persist_false.jsonl")

    # 1. Run when file does not exist
    res1 = run_drift_report(failed_df.iloc[:50], persist=False, log_path=test_log)
    assert "system_status" in res1
    assert not os.path.exists(test_log)

    # 2. Run when file already exists
    with open(test_log, "w", encoding="utf-8") as f:
        f.write('{"existing": "data"}\n')

    res2 = run_drift_report(failed_df.iloc[:50], persist=False, log_path=test_log)
    assert "system_status" in res2

    with open(test_log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # File contents remain completely unchanged!
    assert len(lines) == 1
    assert "existing" in lines[0]
