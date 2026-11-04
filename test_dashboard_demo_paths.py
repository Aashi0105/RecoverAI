"""
Empirical test verifying the 3 critical decision paths (ACT, ESCALATE, REFUSE)
and end-to-end orchestration workflow.
"""
import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent.orchestrator import orchestrate_transaction
from frontend.app import load_and_split_dataset, csv_mtime

test_df, idx_dev, idx_test = load_and_split_dataset(csv_mtime=csv_mtime)

print("=" * 95)
print(" 🔬 RECOVERAI DEMO PATHS & WORKFLOW VERIFICATION")
print("=" * 95)

# Dynamically locate representative candidate transactions for ACT, ESCALATE, and REFUSE
act_row = None
esc_row = None
ref_row = None

for _, row in test_df.iterrows():
    r_dict = row.to_dict()
    res = orchestrate_transaction(r_dict, dry_run=True)
    dec = res["safety"]["decision"]
    if dec == "ACT" and act_row is None:
        act_row = r_dict
    elif dec == "ESCALATE" and esc_row is None:
        esc_row = r_dict
    elif dec == "REFUSE" and ref_row is None:
        ref_row = r_dict
    if act_row and esc_row and ref_row:
        break

# A. ACT PATH
res_act = orchestrate_transaction(act_row, dry_run=True)
print(f"\n[A. ACT PATH — {res_act['transaction_id']}]")
print(f" • Transaction ID        : {res_act['transaction_id']}")
print(f" • Amount                : ₹{res_act['amount']:,.2f}")
print(f" • Recovery Probability  : {res_act['prediction']['recovery_probability']:.4f}")
print(f" • Recommended Action    : {res_act['business_policy']['recommended_action']}")
print(f" • Safety Gate Decision  : {res_act['safety']['decision']}")
print(f" • Execution Status      : {res_act['execution']['execution_status']}")

assert res_act["safety"]["decision"] == "ACT"
assert res_act["execution"]["execution_status"] in ["SIMULATED_DRY_RUN", "SIMULATED_SUCCESS_DRY_RUN", "SUCCEEDED"]

# B. ESCALATE PATH
res_esc = orchestrate_transaction(esc_row, dry_run=True)
print(f"\n[B. ESCALATE PATH — {res_esc['transaction_id']}]")
print(f" • Transaction ID        : {res_esc['transaction_id']}")
print(f" • Amount                : ₹{res_esc['amount']:,.2f}")
print(f" • Recovery Probability  : {res_esc['prediction']['recovery_probability']:.4f}")
print(f" • Recommended Action    : {res_esc['business_policy']['recommended_action']}")
print(f" • Safety Gate Decision  : {res_esc['safety']['decision']}")
print(f" • Triggered Rules       : {res_esc['safety']['triggered_rules']}")
print(f" • Execution Status      : {res_esc['execution']['execution_status']}")

assert res_esc["safety"]["decision"] == "ESCALATE"
assert res_esc["execution"]["execution_status"] in ["BLOCKED_BY_POLICY", "ESCALATED_HUMAN_REVIEW"]

# C. REFUSE PATH
res_ref = orchestrate_transaction(ref_row, dry_run=True)
print(f"\n[C. REFUSE PATH — {res_ref['transaction_id']}]")
print(f" • Transaction ID        : {res_ref['transaction_id']}")
print(f" • Amount                : ₹{res_ref['amount']:,.2f}")
print(f" • Recovery Probability  : {res_ref['prediction']['recovery_probability']:.4f}")
print(f" • Recommended Action    : {res_ref['business_policy']['recommended_action']}")
print(f" • Safety Gate Decision  : {res_ref['safety']['decision']}")
print(f" • Triggered Rules       : {res_ref['safety']['triggered_rules']}")
print(f" • Execution Status      : {res_ref['execution']['execution_status']}")

assert res_ref["safety"]["decision"] == "REFUSE"
assert res_ref["execution"]["execution_status"] in ["BLOCKED_BY_POLICY", "ACTION_REFUSED_BY_POLICY"]

print("\n ✅ ALL 3 CRITICAL DECISION PATHS VERIFIED PERFECTLY!")
print("=" * 95 + "\n")
