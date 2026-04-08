"""Baseline agents for FraudShield."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from models import DecisionEnum, FraudCheckAction

try:  # pragma: no cover - optional in local smoke tests
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency installed in submission image
    OpenAI = None

logger = logging.getLogger(__name__)
DEFAULT_PROXY_MODEL = "gpt-4o-mini"


def get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped_value = value.strip()
            if stripped_value:
                return stripped_value
    return default


class HeuristicFraudDetectionAgent:
    """Deterministic local fallback for offline testing."""

    name = "heuristic-baseline"

    def decide(self, observation) -> FraudCheckAction:
        data = observation.transaction_data
        history = observation.historical_context or {}
        risk_points = 0
        reasons = []

        amount_gap = data.amount / max(data.item_price, 1.0)
        device_mismatch = data.device_country != data.shipping_address

        if data.previous_fraud_flags > 0:
            risk_points += 2
            reasons.append("related accounts were flagged before")
        if data.seller_chargeback_rate_30d >= 0.10:
            risk_points += 2
            reasons.append("seller chargeback rate is elevated")
        if data.shared_device_accounts_24h >= 6:
            risk_points += 2
            reasons.append("device was reused by many accounts")
        if data.same_address_orders_24h >= 5:
            risk_points += 1
            reasons.append("address velocity is unusually high")
        if data.seller_account_age_days <= 30 and amount_gap >= 1.20:
            risk_points += 2
            reasons.append("new seller with a suspicious price gap")
        if device_mismatch:
            risk_points += 1
            reasons.append("device country does not match shipping country")
        if data.buyer_disputes_90d >= 2:
            risk_points += 1
            reasons.append("buyer has recent disputes")
        if history.get("cluster_alert_score", 0.0) >= 0.75:
            risk_points += 1
            reasons.append("network cluster score is high")
        if data.is_repeat_buyer and data.seller_avg_rating >= 4.5:
            risk_points -= 2
            reasons.append("repeat buyer with a highly rated seller")
        if data.num_seller_reviews >= 500 and data.seller_chargeback_rate_30d <= 0.03:
            risk_points -= 1
            reasons.append("seller has strong review and chargeback history")

        threshold = {"easy": 3, "medium": 4, "hard": 5}[observation.task_name.value]
        margin = risk_points - threshold
        decision = DecisionEnum.FRAUD if margin >= 0 else DecisionEnum.LEGITIMATE
        confidence = min(0.95, max(0.55, 0.60 + abs(margin) * 0.08))
        reasoning = "; ".join(reasons[:3]) if reasons else "signals are mixed but skew toward legitimate behavior"

        return FraudCheckAction(
            transaction_id=observation.transaction_id,
            decision=decision,
            confidence=round(confidence, 2),
            reasoning=reasoning[:500],
        )


class OpenAIFraudDetectionAgent:
    """OpenAI-compatible agent used by the competition inference script."""

    name = "openai-client-baseline"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install project dependencies first.")

        self.model_name = model_name
        self.api_base_url = api_base_url or "https://router.huggingface.co/v1"
        self.client = OpenAI(base_url=self.api_base_url, api_key=api_key, timeout=timeout)

    def decide(self, observation) -> FraudCheckAction:
        """Classify the current transaction with an OpenAI-compatible chat model."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                max_tokens=180,
                messages=self._build_messages(observation),
            )
            response_text = completion.choices[0].message.content or ""
            payload = self._parse_payload(response_text)
            decision = DecisionEnum(payload["decision"])
            confidence = float(max(0.0, min(1.0, payload["confidence"])))
            reasoning = str(payload["reasoning"])[:500]
            if len(reasoning) < 10:
                raise ValueError("reasoning is too short")
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
            )
        except Exception as exc:  # pragma: no cover - depends on external API
            raise RuntimeError(
                "OpenAI baseline request failed for "
                f"{observation.transaction_id} using model '{self.model_name}' at '{self.api_base_url}'. "
                "Check API_BASE_URL, API_KEY, and MODEL_NAME."
            ) from exc

    def _build_messages(self, observation) -> list[Dict[str, str]]:
        data = observation.transaction_data
        history = observation.historical_context or {}
        user_prompt = {
            "task": observation.task_name.value,
            "transaction_id": observation.transaction_id,
            "transaction": {
                "amount": data.amount,
                "item_price": data.item_price,
                "item_category": data.item_category,
                "seller_account_age_days": data.seller_account_age_days,
                "buyer_account_age_days": data.buyer_account_age_days,
                "payment_method": data.payment_method,
                "shipping_address": data.shipping_address,
                "device_country": data.device_country,
                "shipping_speed": data.shipping_speed,
                "seller_avg_rating": data.seller_avg_rating,
                "num_seller_reviews": data.num_seller_reviews,
                "previous_fraud_flags": data.previous_fraud_flags,
                "seller_chargeback_rate_30d": data.seller_chargeback_rate_30d,
                "buyer_disputes_90d": data.buyer_disputes_90d,
                "shared_device_accounts_24h": data.shared_device_accounts_24h,
                "same_address_orders_24h": data.same_address_orders_24h,
                "amount_percentile": data.amount_percentile,
                "is_repeat_buyer": data.is_repeat_buyer,
            },
            "historical_context": history,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are reviewing one marketplace transaction for fraud. "
                    "Return only JSON with keys decision, confidence, and reasoning. "
                    "decision must be fraud or legitimate. confidence must be a number between 0 and 1. "
                    "reasoning must be one short sentence grounded in the evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_prompt, separators=(",", ":")),
            },
        ]

    @staticmethod
    def _parse_payload(response_text: str) -> Dict[str, Any]:
        response_text = response_text.strip()
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("model did not return JSON")
        payload = json.loads(response_text[start : end + 1])
        if "decision" not in payload or "confidence" not in payload or "reasoning" not in payload:
            raise ValueError("response is missing required keys")
        return payload


def discover_model_name(api_key: str, api_base_url: str) -> Optional[str]:
    """Query the configured proxy for available models and pick a sensible default."""

    if OpenAI is None:
        raise ImportError("openai package is not installed. Install project dependencies first.")

    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=15.0)

    try:
        response = client.models.list()
        model_ids = sorted(
            {
                getattr(model, "id", "").strip()
                for model in response
                if getattr(model, "id", "").strip()
            }
        )
    except Exception as exc:
        logger.warning(
            "Could not list models from API_BASE_URL=%s: %s. Falling back to default proxy model %s.",
            api_base_url,
            exc,
            DEFAULT_PROXY_MODEL,
        )
        return DEFAULT_PROXY_MODEL

    if not model_ids:
        logger.warning(
            "The proxy at API_BASE_URL=%s returned no models. Falling back to default proxy model %s.",
            api_base_url,
            DEFAULT_PROXY_MODEL,
        )
        return DEFAULT_PROXY_MODEL

    preferred_models = [
        DEFAULT_PROXY_MODEL,
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
    ]
    for preferred_model in preferred_models:
        if preferred_model in model_ids:
            return preferred_model

    return model_ids[0]


def build_default_agent() -> object:
    """Create the required OpenAI client agent when configured, else use the offline fallback."""

    model_name = get_env("MODEL_NAME", "MODELNAME")
    api_key = get_env("API_KEY", "APIKEY", "OPENAI_API_KEY", "OPENAIAPIKEY", "HF_TOKEN", "HFTOKEN")
    api_base_url = get_env("API_BASE_URL", "APIBASEURL")

    if api_key:
        resolved_api_base_url = api_base_url or "https://router.huggingface.co/v1"
        resolved_model_name = model_name or discover_model_name(api_key, resolved_api_base_url)
        try:
            return OpenAIFraudDetectionAgent(
                model_name=resolved_model_name,
                api_key=api_key,
                api_base_url=resolved_api_base_url,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize OpenAI agent: %s. Falling back to heuristic.", exc
            )
            return HeuristicFraudDetectionAgent()

    if model_name and not api_key:
        logger.warning(
            "MODEL_NAME was set but no API_KEY-compatible credential was available. "
            "Falling back to the deterministic heuristic agent."
        )
    else:
        logger.warning(
            "API_KEY-compatible credentials were not set. "
            "Falling back to the deterministic heuristic agent."
        )
    return HeuristicFraudDetectionAgent()
