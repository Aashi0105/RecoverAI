"""
RecoverAI - Metric Comparison Audit Script
Verifies alignment between default industry baseline (tau=0.50) and RecoverAI EV-optimal policy (tau=0.35).
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from verify_canonical_metrics import main

if __name__ == "__main__":
    main()
