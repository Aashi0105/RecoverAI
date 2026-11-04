"""
Empirical verification script ensuring Streamlit UI reruns with persist=False
do NOT increment line counts in logs/drift_audit.jsonl.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from monitoring.drift_detection import run_drift_report, DRIFT_AUDIT_LOG_PATH
from frontend.app import load_and_split_dataset, csv_mtime

test_df, _, _ = load_and_split_dataset(csv_mtime=csv_mtime)

initial_count = 0
if os.path.exists(DRIFT_AUDIT_LOG_PATH):
    with open(DRIFT_AUDIT_LOG_PATH, "r") as f:
        initial_count = len(f.readlines())

# Simulate 10 Streamlit UI reruns with persist=False (what frontend/app.py now calls)
for _ in range(10):
    run_drift_report(test_df, persist=False)

final_count = 0
if os.path.exists(DRIFT_AUDIT_LOG_PATH):
    with open(DRIFT_AUDIT_LOG_PATH, "r") as f:
        final_count = len(f.readlines())

print("=" * 85)
print(" 🔬 STREAMLIT DRIFT DUPLICATION FIX VERIFICATION")
print("=" * 85)
print(f" • Initial JSONL Line Count     : {initial_count}")
print(f" • Simulated 10 Reruns (persist=False)")
print(f" • Final JSONL Line Count       : {final_count}")
print(f" • Line Count Increase          : {final_count - initial_count}")

assert final_count == initial_count

# Now test intentional persistence run (persist=True)
run_drift_report(test_df, persist=True)
count_after_intentional = 0
if os.path.exists(DRIFT_AUDIT_LOG_PATH):
    with open(DRIFT_AUDIT_LOG_PATH, "r") as f:
        count_after_intentional = len(f.readlines())

print(f" • Intentional Run (persist=True) Line Count : {count_after_intentional}")
assert count_after_intentional == initial_count + 1

print("\n ✅ STREAMLIT DRIFT AUDIT DUPLICATION FIX VERIFIED PERFECTLY!")
print("=" * 85 + "\n")
