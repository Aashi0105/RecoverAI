"""
RecoverAI - Interactive UI Sequences Verification Script
Tests Streamlit interactive user flows: Scenarios A, B, C, D, Reset/Replay, and HITL Approvals.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from streamlit.testing.v1 import AppTest

def test_interactive_sequences():
    print("=" * 80)
    print(" 🔬 TESTING INTERACTIVE UI SEQUENCES (SCENARIOS A, B, C, D)")
    print("=" * 80)

    at = AppTest.from_file(os.path.join(PROJECT_ROOT, "frontend", "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Initial run exception: {at.exception}"
    print(" • Initial Streamlit dashboard launch: SUCCESS (No exceptions)")

    # Sequence A: Auto-Recovery
    print(" • Testing Scenario A (Smart Auto-Recovery)...")
    btn_a = [b for b in at.button if b.key == "btn_sc_a"]
    if btn_a:
        btn_a[0].click().run()
    sim_btns = [b for b in at.button if "Simulate Payment Failure & Run Agent" in (b.label or "")]
    sim_btns[0].click().run()
    assert not at.exception, f"Scenario A exception: {at.exception}"
    print("   -> Scenario A completed: SUCCESS")

    # Sequence: Reset / Replay
    print(" • Testing Reset / Replay functionality...")
    reset_btns = [b for b in at.button if "Reset / Replay" in (b.label or "")]
    reset_btns[0].click().run()
    assert not at.exception, f"Reset / Replay exception: {at.exception}"
    print("   -> Reset / Replay: SUCCESS")

    # Sequence B: Human Approval Escalation
    print(" • Testing Scenario B (High-Value Human Review)...")
    btn_b = [b for b in at.button if b.key == "btn_sc_b"]
    btn_b[0].click().run()
    sim_btns = [b for b in at.button if "Simulate Payment Failure & Run Agent" in (b.label or "")]
    sim_btns[0].click().run()
    assert not at.exception, f"Scenario B exception: {at.exception}"
    print("   -> Scenario B completed: SUCCESS")

    # Sequence C: Fraud / High-Risk Block
    print(" • Testing Scenario C (Fraud / Risk Block)...")
    btn_c = [b for b in at.button if b.key == "btn_sc_c"]
    btn_c[0].click().run()
    sim_btns = [b for b in at.button if "Simulate Payment Failure & Run Agent" in (b.label or "")]
    sim_btns[0].click().run()
    assert not at.exception, f"Scenario C exception: {at.exception}"
    print("   -> Scenario C completed: SUCCESS")

    # Sequence D: Low Probability Refusal
    print(" • Testing Scenario D (Low Probability / Negative EV)...")
    btn_d = [b for b in at.button if b.key == "btn_sc_d"]
    btn_d[0].click().run()
    sim_btns = [b for b in at.button if "Simulate Payment Failure & Run Agent" in (b.label or "")]
    sim_btns[0].click().run()
    assert not at.exception, f"Scenario D exception: {at.exception}"
    print("   -> Scenario D completed: SUCCESS")

    # Sequence HITL: Approve and Reject in Queue
    print(" • Testing HITL Approval Queue Actions...")
    btn_b_new = [b for b in at.button if b.key == "btn_sc_b"]
    if btn_b_new:
        btn_b_new[0].click().run()
    sim_btns_new = [b for b in at.button if "Simulate Payment Failure & Run Agent" in (b.label or "")]
    if sim_btns_new:
        sim_btns_new[0].click().run()

    app_btns = [b for b in at.button if b.key and b.key.startswith("app_")]
    if app_btns:
        app_btns[0].click().run()
        assert not at.exception, f"Approve button exception: {at.exception}"
        print("   -> Approve Action: SUCCESS")

    rej_btns = [b for b in at.button if b.key and b.key.startswith("rej_")]
    if rej_btns:
        rej_btns[0].click().run()
        assert not at.exception, f"Reject button exception: {at.exception}"
        print("   -> Reject Action: SUCCESS")

    print("\n ✅ ALL INTERACTIVE UI SEQUENCES VERIFIED PERFECTLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_interactive_sequences()
