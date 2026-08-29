"""
Node 1: Load Context
Prepares structured pre-recovery context for the agent workflow.
"""

from agent.state import AgentState


def load_context(state: AgentState) -> AgentState:
    """
    Extracts and structures pre-decision context from input failed transaction.
    """
    amount = float(state.get("amount", 0.0))
    currency = str(state.get("currency", "INR"))
    method = str(state.get("payment_method", "card"))
    network = str(state.get("payment_network", "visa"))
    channel = str(state.get("payment_channel", "web"))
    
    reason = str(state.get("failure_reason", "unknown"))
    category = str(state.get("failure_category", "unknown"))
    
    # Customer history details
    customer_context = {
        "customer_age_days": state.get("customer_age_days", 30),
        "customer_previous_transactions": state.get("customer_previous_transactions", 0),
        "customer_successful_transactions": state.get("customer_successful_transactions", 0),
        "customer_historical_success_rate": state.get("customer_historical_success_rate", 1.0),
        "customer_lifetime_value": state.get("customer_lifetime_value", 0.0),
        "customer_average_transaction": state.get("customer_average_transaction", amount),
        "customer_transaction_frequency_30d": state.get("customer_transaction_frequency_30d", 3),
        "amount_vs_customer_average": state.get("amount_vs_customer_average", 1.0),
        "hour": state.get("hour", 14),
        "day_of_week": state.get("day_of_week", 2),
        "is_weekend": state.get("is_weekend", 0),
        "is_subscription": state.get("is_subscription", 0),
        "subscription_age_days": state.get("subscription_age_days", 0),
        "is_first_transaction": state.get("is_first_transaction", 0),
        "previous_failures_24h": state.get("previous_failures_24h", 0),
        "previous_failures_7d": state.get("previous_failures_7d", 0),
        "previous_successes_30d": state.get("previous_successes_30d", 0),
        "transactions_24h": state.get("transactions_24h", 1),
        "transactions_7d": state.get("transactions_7d", 3),
        "recovery_attempt_count": state.get("recovery_attempt_count", 0),
        "customer_contacted_today": state.get("customer_contacted_today", 0),
        "ip_risk_score": state.get("ip_risk_score", 0.0),
        "velocity_score": state.get("velocity_score", 0.0),
        "device_changed": state.get("device_changed", 0),
        "location_changed": state.get("location_changed", 0)
    }
    
    state["amount"] = amount
    state["currency"] = currency
    state["payment_method"] = method
    state["payment_network"] = network
    state["payment_channel"] = channel
    state["failure_reason"] = reason
    state["failure_category"] = category
    state["recovery_attempt_count"] = state.get("recovery_attempt_count", 0)
    state["customer_contacted_today"] = state.get("customer_contacted_today", 0)
    state["previous_failures_24h"] = state.get("previous_failures_24h", 0)
    state["previous_failures_7d"] = state.get("previous_failures_7d", 0)
    state["ip_risk_score"] = state.get("ip_risk_score", 0.0)
    state["velocity_score"] = state.get("velocity_score", 0.0)
    state["customer_context"] = customer_context
    state["agent_status"] = "ANALYZING"

    return state
