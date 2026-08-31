"""
Reusable Test & Demo Transaction Factory for RecoverAI Agent.

Maintains backward compatibility for demo scripts and unit tests by exposing
build_test_transaction, which delegates to agent.normalizer.normalize_transaction_payload.
"""

from agent.normalizer import normalize_transaction_payload

# Backward compatibility alias
build_test_transaction = normalize_transaction_payload
