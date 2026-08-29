"""
Node 2: ML Recovery Prediction
Queries trained ML model artifact to get P(recovery_success) and calculates Expected Recovery Value.
"""

from agent.state import AgentState
from ml.predict import predict_recovery_probability
from ml.features import APPROVED_MODEL_FEATURES


def predict_recovery(state: AgentState) -> AgentState:
    """
    Computes recovery probability and expected monetary value independently of the LLM.
    """
    # Call ML inference engine (flatten state and customer_context for ML contract)
    input_dict = dict(state)
    if "customer_context" in state and isinstance(state["customer_context"], dict):
        input_dict.update(state["customer_context"])

    ml_input = {
        key: input_dict[key]
        for key in APPROVED_MODEL_FEATURES
    }

    res = predict_recovery_probability(ml_input)
    prob = float(res["recovery_probability"])
    
    amount = float(state.get("amount", 0.0))
    expected_val = round(amount * prob, 2)
    
    ip_risk = float(state.get("customer_context", {}).get("ip_risk_score", 0.0))
    velocity = float(state.get("customer_context", {}).get("velocity_score", 0.0))
    
    if ip_risk > 0.6 or velocity > 0.6:
        risk_level = "HIGH"
    elif ip_risk > 0.3 or velocity > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    state["recovery_probability"] = prob
    state["expected_recovery_value"] = expected_val
    state["risk_level"] = risk_level

    return state
