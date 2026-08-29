"""
Synthetic Dataset Generator for RecoverAI Payment Recovery System.

Generates realistic merchant payment transactions, failure contexts, customer profiles,
and simulated payment recovery outcomes for machine learning model training.
"""

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Allowed Categorical Values
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
PAYMENT_NETWORKS = {
    "card": ["visa", "mastercard", "rupay", "amex"],
    "upi": ["upi"],
    "netbanking": ["hdfc", "icici", "sbi", "axis", "other"],
    "wallet": ["other"]
}
PAYMENT_CHANNELS = ["web", "mobile", "api"]

# Failure mapping: reason -> category
FAILURE_MAPPING = {
    "network_timeout": "transient",
    "technical_error": "technical",
    "insufficient_funds": "customer_action_required",
    "authentication_failed": "customer_action_required",
    "limit_exceeded": "customer_action_required",
    "card_expired": "payment_method_problem",
    "invalid_card": "payment_method_problem",
    "bank_declined": "bank_decline",
    "suspected_risk": "risk_related",
    "customer_cancelled": "customer_action_required"
}

FAILURE_REASONS = list(FAILURE_MAPPING.keys())
FAILURE_CATEGORIES = sorted(list(set(FAILURE_MAPPING.values())))
RECOVERY_ACTIONS = ["none", "retry", "payment_link", "reminder", "escalate", "no_action"]


def generate_customers(num_customers: int = 3000, seed: int = 42) -> list:
    """Generate synthetic customer base with realistic baseline profiles."""
    rng = np.random.default_rng(seed)
    customers = []
    
    for i in range(num_customers):
        cust_id = f"cust_{i+1:05d}"
        age_days = int(rng.integers(1, 1095))  # 1 day to 3 years
        
        # Base tendency for success
        base_success_propensity = float(rng.beta(8, 2))  # generally high (0.7-0.95)
        preferred_method = str(rng.choice(PAYMENT_METHODS, p=[0.45, 0.35, 0.15, 0.05]))
        avg_txn_amount = float(np.round(rng.lognormal(mean=7.5, sigma=0.8), 2))  # ~ INR 1000 to 10,000
        avg_txn_amount = float(np.clip(avg_txn_amount, 100.0, 50000.0))
        
        customers.append({
            "customer_id": cust_id,
            "customer_age_days": age_days,
            "base_success_propensity": base_success_propensity,
            "preferred_method": preferred_method,
            "avg_txn_amount": avg_txn_amount,
            # State tracked across timeline
            "past_txns": 0,
            "past_successes": 0,
            "lifetime_value": 0.0,
            "recent_timestamps": [],
            "recent_failures": [],
            "consecutive_failure_streak": 0,
            "past_recovery_attempts": 0,
            "past_recoveries_successful": 0
        })
    return customers


def generate_merchants(num_merchants: int = 50, seed: int = 42) -> list:
    """Generate synthetic merchant base with industry variations."""
    rng = np.random.default_rng(seed)
    merchants = []
    
    for i in range(num_merchants):
        m_id = f"merch_{i+1:03d}"
        merchant_failure_bias = float(rng.uniform(-0.05, 0.10))
        merchants.append({
            "merchant_id": m_id,
            "failure_bias": merchant_failure_bias
        })
    return merchants


def generate_transactions(
    num_rows: int = 20000,
    customers: list = None,
    merchants: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic sequence of payment transactions."""
    if customers is None:
        customers = generate_customers(num_customers=4000, seed=seed)
    if merchants is None:
        merchants = generate_merchants(num_merchants=50, seed=seed)
        
    rng = np.random.default_rng(seed)
    
    start_date = datetime(2026, 1, 1)
    # Generate sorted timestamps over 90 days
    seconds_offset = np.sort(rng.integers(0, 90 * 86400, size=num_rows))
    
    records = []
    
    for idx in range(num_rows):
        txn_id = f"txn_{idx+1:07d}"
        cust = customers[int(rng.integers(0, len(customers)))]
        merch = merchants[int(rng.integers(0, len(merchants)))]
        
        txn_time = start_date + timedelta(seconds=int(seconds_offset[idx]))
        hour = txn_time.hour
        day_of_week = txn_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Payment details
        method = cust["preferred_method"] if rng.random() < 0.8 else str(rng.choice(PAYMENT_METHODS))
        possible_networks = PAYMENT_NETWORKS[method]
        network = str(rng.choice(possible_networks))
        channel = str(rng.choice(PAYMENT_CHANNELS, p=[0.4, 0.5, 0.1]))
        
        # Transaction Amount
        amount_factor = float(rng.lognormal(mean=0, sigma=0.3))
        amount = round(float(cust["avg_txn_amount"] * amount_factor), 2)
        amount = float(np.clip(amount, 50.0, 100000.0))
        
        # Subscription context
        is_subscription = int(rng.choice([0, 1], p=[0.7, 0.3]))
        subscription_age_days = int(rng.integers(30, 730)) if is_subscription else 0
        
        # Customer history metrics at this time
        past_txns = cust["past_txns"]
        past_successes = cust["past_successes"]
        is_first_transaction = 1 if past_txns == 0 else 0
        hist_success_rate = round(past_successes / past_txns, 4) if past_txns > 0 else 1.0
        ltv = round(cust["lifetime_value"], 2)
        cust_avg_txn = round(ltv / past_successes, 2) if past_successes > 0 else cust["avg_txn_amount"]
        amount_vs_avg = round(amount / (cust_avg_txn + 1e-5), 4)
        
        # Chronological customer recovery rate & failure streak before current transaction
        consecutive_streak = cust["consecutive_failure_streak"]
        past_rec_att = cust["past_recovery_attempts"]
        past_rec_succ = cust["past_recoveries_successful"]
        cust_past_recovery_rate = round(past_rec_succ / past_rec_att, 4) if past_rec_att > 0 else 0.50
        
        # Recent sliding window activity (24h, 7d, 30d)
        cutoff_24h = txn_time - timedelta(hours=24)
        cutoff_7d = txn_time - timedelta(days=7)
        cutoff_30d = txn_time - timedelta(days=30)
        
        recent_ts = cust["recent_timestamps"]
        txns_24h = sum(1 for t in recent_ts if t >= cutoff_24h)
        txns_7d = sum(1 for t in recent_ts if t >= cutoff_7d)
        txns_30d = sum(1 for t in recent_ts if t >= cutoff_30d)
        
        recent_fails = cust["recent_failures"]
        prev_fails_24h = sum(1 for t, status in recent_fails if t >= cutoff_24h and not status)
        prev_fails_7d = sum(1 for t, status in recent_fails if t >= cutoff_7d and not status)
        prev_successes_30d = sum(1 for t, status in recent_fails if t >= cutoff_30d and status)
        
        # Risk / Behavioral signals
        device_changed = int(rng.choice([0, 1], p=[0.92, 0.08]))
        location_changed = int(rng.choice([0, 1], p=[0.88, 0.12]))
        ip_risk_score = round(float(rng.beta(1.5, 8)), 4)
        velocity_score = round(float(np.clip((txns_24h * 0.25) + (prev_fails_24h * 0.3) + rng.uniform(0, 0.2), 0.0, 1.0)), 4)
        
        record = {
            "transaction_id": txn_id,
            "customer_id": cust["customer_id"],
            "merchant_id": merch["merchant_id"],
            "amount": amount,
            "currency": "INR",
            "payment_method": method,
            "payment_network": network,
            "payment_channel": channel,
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "customer_age_days": cust["customer_age_days"],
            "customer_previous_transactions": past_txns,
            "customer_successful_transactions": past_successes,
            "customer_historical_success_rate": hist_success_rate,
            "customer_lifetime_value": ltv,
            "customer_average_transaction": cust_avg_txn,
            "customer_transaction_frequency_30d": txns_30d,
            "amount_vs_customer_average": amount_vs_avg,
            "is_subscription": is_subscription,
            "subscription_age_days": subscription_age_days,
            "is_first_transaction": is_first_transaction,
            "consecutive_failure_streak": consecutive_streak,
            "customer_past_recovery_rate_pre_current": cust_past_recovery_rate,
            "previous_failures_24h": prev_fails_24h,
            "previous_failures_7d": prev_fails_7d,
            "previous_successes_30d": prev_successes_30d,
            "transactions_24h": txns_24h,
            "transactions_7d": txns_7d,
            "device_changed": device_changed,
            "location_changed": location_changed,
            "ip_risk_score": ip_risk_score,
            "velocity_score": velocity_score,
            # Customer & Merchant metadata passed internally to determine failure
            "_base_success_propensity": cust["base_success_propensity"],
            "_merchant_failure_bias": merch["failure_bias"],
            "_cust_ref": cust,
            "_txn_time": txn_time
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    return df


def assign_failure_information(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign payment_status (success/failed) and realistic failure_reason/failure_category."""
    rng = np.random.default_rng(seed)
    
    statuses = []
    reasons = []
    categories = []
    
    # Reason distribution probabilities for failed transactions
    reason_probs = [
        0.22,  # network_timeout (transient)
        0.15,  # technical_error (technical)
        0.18,  # insufficient_funds (customer_action)
        0.12,  # authentication_failed (customer_action)
        0.08,  # limit_exceeded (customer_action)
        0.08,  # bank_declined (bank_decline)
        0.07,  # card_expired (payment_method)
        0.04,  # invalid_card (payment_method)
        0.04,  # suspected_risk (risk_related)
        0.02   # customer_cancelled (customer_action)
    ]
    
    for _, row in df.iterrows():
        # Base failure probability derived from customer propensity, risk signals, and merchant bias
        p_fail = 1.0 - row["_base_success_propensity"]
        p_fail += row["_merchant_failure_bias"]
        if row["ip_risk_score"] > 0.6:
            p_fail += 0.15
        if row["previous_failures_24h"] > 0:
            p_fail += 0.10
        if row["device_changed"] == 1:
            p_fail += 0.05
            
        p_fail = np.clip(p_fail, 0.12, 0.28)  # Overall target failure rate 15-25%
        
        is_failed = rng.random() < p_fail
        cust = row["_cust_ref"]
        
        if not is_failed:
            status = "success"
            reason = "none"
            category = "none"
            
            # Update customer state for future transaction correlation
            cust["past_txns"] += 1
            cust["past_successes"] += 1
            cust["lifetime_value"] += row["amount"]
            cust["recent_timestamps"].append(row["_txn_time"])
            cust["recent_failures"].append((row["_txn_time"], True))
            cust["consecutive_failure_streak"] = 0  # Reset streak on success
        else:
            status = "failed"
            reason = str(rng.choice(FAILURE_REASONS, p=reason_probs))
            category = FAILURE_MAPPING[reason]
            
            cust["past_txns"] += 1
            cust["recent_timestamps"].append(row["_txn_time"])
            cust["recent_failures"].append((row["_txn_time"], False))
            cust["consecutive_failure_streak"] += 1  # Increment failure streak
            
        statuses.append(status)
        reasons.append(reason)
        categories.append(category)
        
    df["payment_status"] = statuses
    df["failure_reason"] = reasons
    df["failure_category"] = categories
    
    # Drop internal helper references
    df = df.drop(columns=["_base_success_propensity", "_merchant_failure_bias", "_cust_ref", "_txn_time"])
    return df


def calculate_recovery_probability(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Calculate underlying recovery probability for failed payments based on realistic business factors.
    For successful payments, recovery_success_probability is set to NaN.
    """
    rng = np.random.default_rng(seed)
    probabilities = []
    
    for _, row in df.iterrows():
        if row["payment_status"] == "success":
            probabilities.append(np.nan)
            continue
            
        # Base probability for failed payments ~ 0.50
        prob = 0.50
        
        # Category / Reason adjustments
        reason = row["failure_reason"]
        category = row["failure_category"]
        
        if category == "transient":
            prob += 0.25
        elif category == "technical":
            prob += 0.20
        elif category == "customer_action_required":
            if reason == "insufficient_funds":
                prob += 0.15
            elif reason == "authentication_failed":
                prob += 0.15
            else:
                prob += 0.10
        elif category == "bank_decline":
            prob += 0.05
        elif category == "payment_method_problem":
            if reason == "card_expired":
                prob -= 0.25
            else:
                prob -= 0.35
        elif category == "risk_related":
            prob -= 0.35
            
        # Customer history positive factors
        if row["customer_historical_success_rate"] >= 0.85:
            prob += 0.12
        elif row["customer_historical_success_rate"] >= 0.70:
            prob += 0.06
            
        if row["customer_previous_transactions"] >= 5:
            prob += 0.06
            
        if row["is_first_transaction"] == 1:
            prob -= 0.08
            
        # Chronological streak & past recovery tendency
        streak = row.get("consecutive_failure_streak", 0)
        if streak >= 4:
            prob -= 0.22
        elif streak >= 2:
            prob -= 0.12
            
        rec_rate = row.get("customer_past_recovery_rate_pre_current", 0.50)
        if rec_rate >= 0.75:
            prob += 0.12
        elif rec_rate <= 0.25:
            prob -= 0.10
            
        # Recent negative signals & risk interactions
        if row["previous_failures_24h"] >= 2:
            prob -= 0.15
        elif row["previous_failures_24h"] == 1:
            prob -= 0.08
            
        if row["previous_failures_7d"] >= 3:
            prob -= 0.10
            
        if row["ip_risk_score"] > 0.70:
            prob -= 0.12
            
        if row["velocity_score"] > 0.70:
            prob -= 0.08
            
        if row["amount_vs_customer_average"] > 2.5:
            prob -= 0.06
            
        # Non-linear feature interactions
        if category == "transient" and row["amount"] > 5000:
            prob += 0.08  # High-value transient failures are actively retried
        if category == "risk_related" and row["ip_risk_score"] > 0.60:
            prob -= 0.12  # Risk category + high IP risk compounding penalty
            
        # Controlled random noise
        noise = rng.normal(0, 0.06)
        prob = float(np.clip(prob + noise, 0.02, 0.98))
        probabilities.append(round(prob, 4))
        
    df["recovery_success_probability"] = probabilities
    return df


def calculate_treatment_effect(row: pd.Series) -> float:
    """
    Computes deterministic heterogeneous treatment effect delta(X) based strictly on pre-treatment features.
    """
    reason = str(row.get("failure_reason", "none")).lower()
    streak = int(row.get("consecutive_failure_streak", 0))
    ip_risk = float(row.get("ip_risk_score", 0.0))

    if reason == "network_timeout":
        delta = 0.18
    elif reason == "technical_error":
        delta = 0.15
    elif reason == "insufficient_funds":
        delta = 0.12
    elif reason == "authentication_failed":
        delta = 0.10
    elif reason == "limit_exceeded":
        delta = 0.08
    elif reason == "bank_declined":
        delta = 0.05
    elif reason == "customer_cancelled":
        delta = 0.02
    elif reason in ["invalid_card", "card_expired", "suspected_risk"]:
        delta = 0.00
    else:
        delta = 0.05

    if streak >= 3:
        delta -= 0.08
    elif streak == 2:
        delta -= 0.04

    if ip_risk > 0.60:
        delta -= 0.06

    return max(0.0, float(delta))


def generate_recovery_outcomes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate recovery information, operational recovery actions, and final Bernoulli outcome (recovered, recovered_amount).
    Encodes heterogeneous treatment effect of recovery_attempted on recovery probability.
    """
    rng = np.random.default_rng(seed)
    
    recovered_list = []
    recovered_amt_list = []
    attempted_list = []
    action_list = []
    attempt_count_list = []
    contacted_today_list = []
    
    for _, row in df.iterrows():
        if row["payment_status"] == "success":
            recovered_list.append(0)
            recovered_amt_list.append(0.0)
            attempted_list.append(0)
            action_list.append("none")
            attempt_count_list.append(0)
            contacted_today_list.append(0)
            continue
            
        p_base = row["recovery_success_probability"]
        amount = row["amount"]
        category = row["failure_category"]
        
        # 1. Treatment assignment T: 70% treated, 30% control (independent of final outcome)
        attempted = 1 if rng.random() < 0.70 else 0
        attempted_list.append(attempted)
        
        # 2. Calculate pre-treatment feature-dependent treatment effect
        if attempted == 1:
            delta = calculate_treatment_effect(row)
            p_final = float(np.clip(p_base + delta, 0.02, 0.98))
        else:
            p_final = p_base
            
        # 3. Sample final recovery outcome Y ~ Bernoulli(p_final)
        is_recovered = int(rng.binomial(1, p_final))
        recovered_list.append(is_recovered)
        
        if is_recovered == 1:
            if rng.random() < 0.95:
                rec_amt = amount
            else:
                rec_amt = round(amount * float(rng.uniform(0.85, 0.99)), 2)
            recovered_amt_list.append(rec_amt)
        else:
            recovered_amt_list.append(0.0)
            
        # Recovery action & attempt details
        if attempted == 0:
            action = "no_action"
            attempt_count = 0
            contacted_today = 0
        else:
            attempt_count = int(rng.choice([1, 2, 3], p=[0.70, 0.20, 0.10]))
            contacted_today = int(rng.choice([0, 1], p=[0.40, 0.60]))
            
            if category == "transient":
                action = "retry" if attempt_count <= 2 else "reminder"
            elif category in ["customer_action_required", "payment_method_problem"]:
                action = "payment_link" if amount > 2000 else "reminder"
            elif category == "risk_related" or amount > 25000:
                action = "escalate"
            else:
                action = str(rng.choice(["retry", "payment_link", "reminder"], p=[0.4, 0.4, 0.2]))
                
        action_list.append(action)
        attempt_count_list.append(attempt_count)
        contacted_today_list.append(contacted_today)
        
    df["recovery_attempted"] = attempted_list
    df["recovery_action"] = action_list
    df["recovery_attempt_count"] = attempt_count_list
    df["customer_contacted_today"] = contacted_today_list
    df["recovered"] = recovered_list
    df["recovered_amount"] = recovered_amt_list
    
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate data quality, schema integrity, and domain rules."""
    errors = []
    
    # 1. Unique transaction_id
    if df["transaction_id"].nunique() != len(df):
        errors.append(f"Duplicate transaction_ids found: {len(df) - df['transaction_id'].nunique()}")
        
    # 2. Required columns check
    required_cols = [
        "transaction_id", "customer_id", "merchant_id", "amount", "currency",
        "payment_method", "payment_network", "payment_channel", "timestamp",
        "hour", "day_of_week", "is_weekend", "customer_age_days",
        "customer_previous_transactions", "customer_successful_transactions",
        "customer_historical_success_rate", "customer_lifetime_value",
        "customer_average_transaction", "customer_transaction_frequency_30d",
        "amount_vs_customer_average", "is_subscription", "subscription_age_days",
        "is_first_transaction", "previous_failures_24h", "previous_failures_7d",
        "previous_successes_30d", "transactions_24h", "transactions_7d",
        "device_changed", "location_changed", "ip_risk_score", "velocity_score",
        "payment_status", "failure_reason", "failure_category",
        "recovery_attempted", "recovery_action", "recovery_attempt_count",
        "customer_contacted_today", "recovery_success_probability",
        "recovered", "recovered_amount"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            
    # 3. Missing values in feature columns
    feature_cols = [c for c in required_cols if c not in ["recovery_success_probability"]]
    null_counts = df[feature_cols].isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Unexpected missing values in feature columns: {null_counts[null_counts > 0].to_dict()}")
        
    # 4. Valid categorical values
    invalid_methods = set(df["payment_method"]) - set(PAYMENT_METHODS)
    if invalid_methods:
        errors.append(f"Invalid payment_method: {invalid_methods}")
        
    invalid_statuses = set(df["payment_status"]) - {"success", "failed"}
    if invalid_statuses:
        errors.append(f"Invalid payment_status: {invalid_statuses}")
        
    # 5. Amounts > 0
    if (df["amount"] <= 0).any():
        errors.append("Found transaction amounts <= 0")
        
    # 6. Recovery probability range for failed payments
    failed_df = df[df["payment_status"] == "failed"]
    if (failed_df["recovery_success_probability"] < 0.0).any() or (failed_df["recovery_success_probability"] > 1.0).any():
        errors.append("recovery_success_probability out of bounds [0, 1] for failed payments")
        
    # 7. Recovered target 0 or 1
    if not set(df["recovered"].unique()).issubset({0, 1}):
        errors.append(f"Invalid recovered values: {df['recovered'].unique()}")
        
    # 8. Recovered amount rules
    if (df["recovered_amount"] < 0).any():
        errors.append("Found recovered_amount < 0")
    if (df["recovered_amount"] > df["amount"] + 1e-4).any():
        errors.append("Found recovered_amount > transaction amount")
    if ((df["recovered"] == 0) & (df["recovered_amount"] != 0)).any():
        errors.append("Found recovered_amount > 0 when recovered == 0")
        
    # 9. Successful vs Failed consistency
    succ_df = df[df["payment_status"] == "success"]
    if (succ_df["failure_reason"] != "none").any() or (succ_df["failure_category"] != "none").any():
        errors.append("Successful payments have non-none failure reason or category")
    if (failed_df["failure_reason"] == "none").any() or (failed_df["failure_category"] == "none").any():
        errors.append("Failed payments have 'none' failure reason or category")
        
    # 10. Entity repetition (customers and merchants repeat)
    if df["customer_id"].nunique() >= len(df):
        errors.append("Customer IDs do not repeat across transactions")
    if df["merchant_id"].nunique() >= len(df):
        errors.append("Merchant IDs do not repeat across transactions")
        
    if errors:
        print("❌ Dataset Validation Failed with Errors:")
        for err in errors:
            print(f"  - {err}")
        return False
        
    print("✅ Dataset Validation Passed Successfully! All quality rules satisfied.")
    return True


def save_dataset(df: pd.DataFrame, raw_dir: str = "data/raw") -> tuple:
    """Save complete raw dataset and small sample CSV."""
    os.makedirs(raw_dir, exist_ok=True)
    full_path = os.path.join(raw_dir, "transactions.csv")
    sample_path = os.path.join(raw_dir, "transactions_sample.csv")
    
    df.to_csv(full_path, index=False)
    df.head(100).to_csv(sample_path, index=False)
    
    return full_path, sample_path


def print_summary(df: pd.DataFrame):
    """Print a concise summary of the generated dataset."""
    total_txns = len(df)
    succ_txns = (df["payment_status"] == "success").sum()
    failed_txns = (df["payment_status"] == "failed").sum()
    fail_rate = (failed_txns / total_txns) * 100
    
    failed_df = df[df["payment_status"] == "failed"]
    recovered_txns = (failed_df["recovered"] == 1).sum()
    rec_rate = (recovered_txns / failed_txns * 100) if failed_txns > 0 else 0.0
    
    total_val = df["amount"].sum()
    failed_val = failed_df["amount"].sum()
    recovered_val = df["recovered_amount"].sum()
    
    print("\n" + "=" * 60)
    print(" 📊 RECOVERAI SYNTHETIC DATASET SUMMARY REPORT")
    print("=" * 60)
    print(f" Total Transactions              : {total_txns:,}")
    print(f" Successful Transactions         : {succ_txns:,} ({(succ_txns/total_txns)*100:.2f}%)")
    print(f" Failed Transactions             : {failed_txns:,} ({fail_rate:.2f}%)")
    print(f" Unique Customers                : {df['customer_id'].nunique():,}")
    print(f" Unique Merchants                : {df['merchant_id'].nunique():,}")
    print(f" Total Transaction Value         : ₹{total_val:,.2f}")
    print(f" Total Failed Transaction Value  : ₹{failed_val:,.2f}")
    print(f" Total Recovered Value           : ₹{recovered_val:,.2f}")
    print(f" Recovery Rate (among failed)   : {rec_rate:.2f}% ({recovered_txns:,}/{failed_txns:,})")
    print(f" Average Transaction Amount      : ₹{df['amount'].mean():,.2f}")
    print(f" Average Failed Txn Amount       : ₹{failed_df['amount'].mean():,.2f}")
    print("-" * 60)
    
    print("\n📌 Failure Category Distribution (Failed Payments):")
    cat_dist = failed_df["failure_category"].value_counts(normalize=True) * 100
    for cat, pct in cat_dist.items():
        count = (failed_df["failure_category"] == cat).sum()
        print(f"  - {cat:<26}: {count:5d} ({pct:5.2f}%)")
        
    print("\n📌 Payment Method Distribution:")
    pm_dist = df["payment_method"].value_counts(normalize=True) * 100
    for pm, pct in pm_dist.items():
        count = (df["payment_method"] == pm).sum()
        print(f"  - {pm:<26}: {count:5d} ({pct:5.2f}%)")
        
    print("\n📌 Recovery Action Distribution (Failed Payments):")
    act_dist = failed_df["recovery_action"].value_counts(normalize=True) * 100
    for act, pct in act_dist.items():
        count = (failed_df["recovery_action"] == act).sum()
        print(f"  - {act:<26}: {count:5d} ({pct:5.2f}%)")
    print("=" * 60 + "\n")


def compute_chronological_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes consecutive_failure_streak for each customer strictly prior to current transaction.
    Requires sorting by customer_id and timestamp.
    Zero leakage: Current payment_status is used to update state AFTER capturing pre-current streak.
    """
    df_sorted = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)
    streaks = []
    customer_streaks = {}

    for _, row in df_sorted.iterrows():
        cid = row["customer_id"]
        if cid not in customer_streaks:
            customer_streaks[cid] = 0

        # Capture pre-current streak BEFORE updating state
        streaks.append(customer_streaks[cid])

        # Update streak AFTER capturing pre-current feature
        status = row["payment_status"]
        if status == "success":
            customer_streaks[cid] = 0
        elif status == "failed":
            customer_streaks[cid] += 1

    df_sorted["consecutive_failure_streak"] = streaks
    return df_sorted.sort_values(by=["transaction_id"]).reset_index(drop=True)


def compute_chronological_past_recovery_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes customer_past_recovery_rate_pre_current for each customer strictly prior to current transaction.
    Requires sorting by customer_id and timestamp.
    Zero leakage: Current recovery outcome is used to update state AFTER capturing pre-current recovery rate.
    """
    df_sorted = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)
    rec_rates = []
    customer_recovery_stats = {}

    for _, row in df_sorted.iterrows():
        cid = row["customer_id"]
        if cid not in customer_recovery_stats:
            customer_recovery_stats[cid] = {"attempts": 0, "successes": 0}

        stats = customer_recovery_stats[cid]
        attempts = stats["attempts"]
        successes = stats["successes"]

        # Capture pre-current rate BEFORE updating state
        current_rate = round(successes / attempts, 4) if attempts > 0 else 0.50
        rec_rates.append(current_rate)

        # Update recovery stats AFTER capturing pre-current feature
        if row["payment_status"] == "failed":
            att = row.get("recovery_attempted", 0)
            rec = row.get("recovered", 0)
            if att == 1:
                stats["attempts"] += 1
                if rec == 1:
                    stats["successes"] += 1

    df_sorted["customer_past_recovery_rate_pre_current"] = rec_rates
    return df_sorted.sort_values(by=["transaction_id"]).reset_index(drop=True)


def generate_pipeline(rows: int = 20000, seed: int = 42, output_dir: str = "data/raw"):
    """Complete data generation pipeline."""
    print(f"🚀 Generating {rows:,} synthetic payment records (seed={seed})...")
    
    num_cust = max(1000, rows // 5)
    customers = generate_customers(num_customers=num_cust, seed=seed)
    merchants = generate_merchants(num_merchants=50, seed=seed)
    
    df = generate_transactions(num_rows=rows, customers=customers, merchants=merchants, seed=seed)
    df = assign_failure_information(df, seed=seed)
    df = compute_chronological_streaks(df)
    df = calculate_recovery_probability(df, seed=seed)
    df = generate_recovery_outcomes(df, seed=seed)
    df = compute_chronological_past_recovery_rates(df)
    
    is_valid = validate_dataset(df)
    if not is_valid:
        raise ValueError("Generated dataset failed validation rules.")
        
    full_path, sample_path = save_dataset(df, raw_dir=output_dir)
    print(f"📁 Dataset saved to: {full_path}")
    print(f"📄 Sample saved to: {sample_path}")
    
    print_summary(df)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecoverAI Synthetic Dataset Generator")
    parser.add_argument("--rows", type=int, default=20000, help="Number of transaction records to generate (default: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory path (default: data/raw)")
    
    args = parser.parse_args()
    generate_pipeline(rows=args.rows, seed=args.seed, output_dir=args.output_dir)
