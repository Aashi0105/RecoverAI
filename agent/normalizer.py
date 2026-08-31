"""
Transaction Normalizer & Context Builder for RecoverAI.

Guarantees that all incoming transaction payloads are normalized to contain
all APPROVED_MODEL_FEATURES required by the ML model inference contract,
along with agent/policy operational fields.
"""

from typing import Dict, Any
from ml.features import APPROVED_MODEL_FEATURES


def normalize_transaction_payload(**kwargs) -> Dict[str, Any]:
    """
    Normalizes an incoming transaction dictionary, filling in defaults for
    optional customer/risk context fields so the payload strictly conforms
    to the 31 APPROVED_MODEL_FEATURES contract and operational fields.
    """
    # 1. Base operational & context defaults
    txn_id = str(kwargs.get("transaction_id", "txn_test_00001"))
    cust_id = str(kwargs.get("customer_id", "cust_00001"))
    merch_id = str(kwargs.get("merchant_id", "merch_001"))
    currency = str(kwargs.get("currency", "INR"))
    
    amount = float(kwargs.get("amount", 2500.0))
    cust_avg_txn = float(kwargs.get("customer_average_transaction", 2500.0))
    amount_vs_avg = round(amount / (cust_avg_txn + 1e-5), 4) if "amount_vs_customer_average" not in kwargs else float(kwargs["amount_vs_customer_average"])
    
    # 2. Time defaults
    hour = int(kwargs.get("hour", 14))
    day_of_week = int(kwargs.get("day_of_week", 2))
    is_weekend = int(kwargs.get("is_weekend", 1 if day_of_week >= 5 else 0))
    
    # 3. Payment method & channel
    payment_method = str(kwargs.get("payment_method", "card"))
    payment_network = str(kwargs.get("payment_network", "visa"))
    payment_channel = str(kwargs.get("payment_channel", "web"))
    
    # 4. Failure information
    failure_category = str(kwargs.get("failure_category", "transient"))
    failure_reason = str(kwargs.get("failure_reason", "network_timeout"))
    
    # 5. Customer history & counts
    cust_age_days = int(kwargs.get("customer_age_days", 180))
    past_txns = int(kwargs.get("customer_previous_transactions", 10))
    is_first_txn = 1 if past_txns == 0 else int(kwargs.get("is_first_transaction", 0))
    
    past_successes = int(kwargs.get("customer_successful_transactions", min(8, past_txns)))
    hist_success_rate = round(past_successes / past_txns, 4) if past_txns > 0 else 1.0
    if "customer_historical_success_rate" in kwargs:
        hist_success_rate = float(kwargs["customer_historical_success_rate"])
        
    ltv = float(kwargs.get("customer_lifetime_value", round(past_successes * cust_avg_txn, 2)))
    freq_30d = int(kwargs.get("customer_transaction_frequency_30d", 3))
    
    # 6. Subscription context
    is_sub = int(kwargs.get("is_subscription", 1))
    sub_age = int(kwargs.get("subscription_age_days", 90)) if is_sub else 0
    
    # 7. Recent failure & transaction activity
    prev_fails_24h = int(kwargs.get("previous_failures_24h", 0))
    prev_fails_7d = int(kwargs.get("previous_failures_7d", max(prev_fails_24h, kwargs.get("previous_failures_7d", 0))))
    prev_succ_30d = int(kwargs.get("previous_successes_30d", 3))
    txns_24h = int(kwargs.get("transactions_24h", 1))
    txns_7d = int(kwargs.get("transactions_7d", max(txns_24h, kwargs.get("transactions_7d", 3))))
    streak = int(kwargs.get("consecutive_failure_streak", prev_fails_24h))
    past_rec_rate = float(kwargs.get("customer_past_recovery_rate_pre_current", 0.0))

    # 8. Behavioral & Risk signals
    device_changed = int(kwargs.get("device_changed", 0))
    location_changed = int(kwargs.get("location_changed", 0))
    ip_risk = float(kwargs.get("ip_risk_score", 0.08))
    velocity = float(kwargs.get("velocity_score", 0.12))

    # 9. Operational agent/policy fields
    attempt_count = int(kwargs.get("recovery_attempt_count", prev_fails_24h))
    contacted_today = int(kwargs.get("customer_contacted_today", 0))

    # Assemble base record
    record = {
        # Operational Identifiers
        "transaction_id": txn_id,
        "customer_id": cust_id,
        "merchant_id": merch_id,
        "currency": currency,
        "recovery_attempt_count": attempt_count,
        "customer_contacted_today": contacted_today,
        "payment_status": "failed",

        # 31 APPROVED ML FEATURES
        "amount": amount,
        "amount_vs_customer_average": amount_vs_avg,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_subscription": is_sub,
        "subscription_age_days": sub_age,
        "is_first_transaction": is_first_txn,
        "customer_age_days": cust_age_days,
        "customer_previous_transactions": past_txns,
        "customer_successful_transactions": past_successes,
        "customer_historical_success_rate": hist_success_rate,
        "customer_lifetime_value": ltv,
        "customer_average_transaction": cust_avg_txn,
        "customer_transaction_frequency_30d": freq_30d,
        "consecutive_failure_streak": streak,
        "customer_past_recovery_rate_pre_current": past_rec_rate,
        "previous_failures_24h": prev_fails_24h,
        "previous_failures_7d": prev_fails_7d,
        "previous_successes_30d": prev_succ_30d,
        "transactions_24h": txns_24h,
        "transactions_7d": txns_7d,
        "device_changed": device_changed,
        "location_changed": location_changed,
        "ip_risk_score": ip_risk,
        "velocity_score": velocity,
        "payment_method": payment_method,
        "payment_network": payment_network,
        "payment_channel": payment_channel,
        "failure_reason": failure_reason,
        "failure_category": failure_category
    }

    # Verify all APPROVED_MODEL_FEATURES exist in record
    missing_feats = [f for f in APPROVED_MODEL_FEATURES if f not in record]
    if missing_feats:
        raise ValueError(f"Normalizer failed to provide required ML features: {missing_feats}")

    return record
