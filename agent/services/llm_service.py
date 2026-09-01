"""
RecoverAI LLM Service Abstraction Layer.

Provides structured decision support for failure diagnosis and recovery recommendation
using OpenAI/Groq compatible chat completion endpoints with Pydantic validation and
fail-safe heuristic fallback.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {"retry", "payment_link", "reminder", "escalate", "no_action"}
ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}


class DiagnosisOutput(BaseModel):
    """Structured output schema for LLM failure diagnosis."""
    failure_category: str
    diagnosis: str
    severity: str = "MEDIUM"
    customer_action_required: bool = False
    key_factors: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)

    @field_validator("severity", mode="before")
    def validate_severity(cls, v):
        val = str(v).upper() if v else "MEDIUM"
        if val not in ALLOWED_SEVERITIES:
            return "MEDIUM"
        return val


class RecommendationOutput(BaseModel):
    """Structured output schema for LLM recovery strategy recommendation."""
    recommended_action: str
    decision_rationale: str
    key_factors: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.80)
    expected_benefit: Optional[str] = None

    @field_validator("recommended_action", mode="before")
    def validate_action(cls, v):
        val = str(v).lower().strip() if v else ""
        if val not in ALLOWED_ACTIONS:
            raise ValueError(f"Action '{val}' is not one of allowed actions: {ALLOWED_ACTIONS}")
        return val


def is_llm_available() -> bool:
    """
    Checks whether LLM mode is enabled and an API key is present.
    Returns False immediately if LLM_ENABLED is false or LLM_API_KEY is empty.
    """
    enabled_env = os.getenv("LLM_ENABLED")
    is_enabled = (enabled_env.lower() == "true") if enabled_env is not None else settings.LLM_ENABLED
    if not is_enabled:
        return False

    api_key = os.getenv("LLM_API_KEY") or settings.LLM_API_KEY
    return bool(api_key and api_key.strip() and not api_key.startswith("your_"))


def get_llm_endpoint_and_headers() -> tuple[str, str, dict]:
    """Resolves API endpoint URL, model name, and authorization headers."""
    provider = (os.getenv("LLM_PROVIDER") or settings.LLM_PROVIDER or "groq").lower()
    api_key = os.getenv("LLM_API_KEY") or settings.LLM_API_KEY
    model = os.getenv("LLM_MODEL") or settings.LLM_MODEL or "llama-3.3-70b-versatile"
    custom_url = os.getenv("LLM_BASE_URL") or settings.LLM_BASE_URL

    if custom_url:
        endpoint = custom_url.rstrip("/") + "/chat/completions"
    elif provider == "groq":
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
    elif provider == "openai":
        endpoint = "https://api.openai.com/v1/chat/completions"
    else:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    return endpoint, model, headers


def call_llm_json(messages: List[Dict[str, str]], timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """
    Executes raw HTTP chat completion request expecting JSON output.
    Returns parsed dictionary or None on any error.
    """
    if not is_llm_available():
        return None

    endpoint, model, headers = get_llm_endpoint_and_headers()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                logger.warning(f"LLM provider returned HTTP {response.status_code}: {response.text[:200]}")
                return None

            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return json.loads(raw_content)
    except Exception as exc:
        logger.warning(f"LLM invocation failed ({type(exc).__name__}: {exc}). Safely falling back to heuristics.")
        return None


def generate_llm_diagnosis(context: Dict[str, Any]) -> Optional[DiagnosisOutput]:
    """
    Generates structured failure diagnosis via LLM.
    Returns validated DiagnosisOutput or None if LLM is unavailable or fails.
    """
    if not is_llm_available():
        return None

    system_prompt = (
        "You are RecoverAI's Payment Failure Diagnostic Engine for Razorpay merchants.\n"
        "Analyze the payment failure context and return a concise, structured JSON object with:\n"
        "- failure_category: str (e.g. transient, technical, customer_action_required, payment_method_problem, bank_decline, risk_related)\n"
        "- diagnosis: str (concise 1-2 sentence explanation for the merchant)\n"
        "- severity: 'LOW' | 'MEDIUM' | 'HIGH'\n"
        "- customer_action_required: bool\n"
        "- key_factors: list of short observable contributing factors\n"
        "- confidence: float between 0.0 and 1.0\n"
        "Do NOT include chain-of-thought or hidden reasoning. Return strictly valid JSON."
    )

    user_prompt = f"Payment failure context:\n{json.dumps(context, indent=2)}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_json = call_llm_json(messages)
    if not raw_json:
        return None

    try:
        return DiagnosisOutput.model_validate(raw_json)
    except Exception as exc:
        logger.warning(f"Diagnosis schema validation failed ({exc}). Falling back to heuristic.")
        return None


def generate_llm_recommendation(context: Dict[str, Any]) -> Optional[RecommendationOutput]:
    """
    Generates structured recovery strategy recommendation via LLM.
    Returns validated RecommendationOutput or None if LLM is unavailable or fails.
    """
    if not is_llm_available():
        return None

    system_prompt = (
        "You are RecoverAI's Revenue Recovery Strategy Engine for Razorpay merchants.\n"
        "Recommend the single most economically viable recovery action based on transaction context, "
        "ML recovery probability, and customer risk factors.\n"
        "ALLOWED ACTIONS (You MUST choose exactly one of these):\n"
        "- 'retry': Automated immediate/background retry (transient network timeouts with low attempt count)\n"
        "- 'payment_link': Create Razorpay payment link for customer to pay via alternative method (UPI, cards)\n"
        "- 'reminder': Gentle customer notification nudge for low-value customer-action failures\n"
        "- 'escalate': Flag for human merchant review (high transaction value or boundary risk)\n"
        "- 'no_action': Abandon recovery to prevent customer annoyance or fraud cost\n\n"
        "Return a strictly valid JSON object with:\n"
        "- recommended_action: str (one of the 5 allowed actions above)\n"
        "- decision_rationale: str (concise 1-2 sentence business rationale)\n"
        "- key_factors: list of short strings (e.g. ['transient network error', 'high ML probability 85%'])\n"
        "- confidence: float between 0.0 and 1.0\n"
        "- expected_benefit: str (expected financial/customer benefit)\n"
        "Do NOT include private chain-of-thought. Return strictly valid JSON."
    )

    user_prompt = f"Recovery context:\n{json.dumps(context, indent=2)}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_json = call_llm_json(messages)
    if not raw_json:
        return None

    try:
        return RecommendationOutput.model_validate(raw_json)
    except Exception as exc:
        logger.warning(f"Recommendation schema validation failed ({exc}). Falling back to heuristic.")
        return None
