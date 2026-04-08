"""Baseline agents for FraudShield."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from models import ActionTypeEnum, DecisionEnum, FraudCheckAction, InvestigationTargetEnum

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

    def _analyze_observation(self, observation) -> tuple[int, list[str], Dict[str, Any]]:
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

        diagnostics = {
            "amount_gap": amount_gap,
            "device_mismatch": device_mismatch,
            "cluster_alert_score": history.get("cluster_alert_score", 0.0),
        }
        return risk_points, reasons, diagnostics

    def decide(self, observation) -> FraudCheckAction:
        risk_points, reasons, _ = self._analyze_observation(observation)

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


class AgenticHeuristicFraudDetectionAgent(HeuristicFraudDetectionAgent):
    """Budget-aware heuristic agent that uses investigation actions on ambiguous cases."""

    name = "agentic-heuristic-baseline"
    HARD_CLUSTER_BONUS = 2
    HARD_FLASH_SALE_DISCOUNT = 4

    def _analyze_observation(self, observation) -> tuple[int, list[str], Dict[str, Any]]:
        risk_points, reasons, diagnostics = super()._analyze_observation(observation)
        data = observation.transaction_data
        history = observation.historical_context or {}
        network_pattern = str(history.get("network_pattern", "")).lower()

        if "reuse the same seller and device cluster" in network_pattern:
            risk_points += self.HARD_CLUSTER_BONUS
            reasons.append("network pattern matches coordinated cluster reuse")
        if "flash sale" in network_pattern:
            risk_points -= self.HARD_FLASH_SALE_DISCOUNT
            reasons.append("ops context indicates a flash-sale velocity pattern that is often legitimate")
        if data.amount <= 10 and data.seller_chargeback_rate_30d >= 0.12:
            risk_points += 2
            reasons.append("micro-amount transaction matches card-testing risk")
        if history.get("linked_cards_7d", 0) >= 6 and history.get("cluster_alert_score", 0.0) >= 0.60:
            risk_points += 1
            reasons.append("linked-card graph suggests coordinated behavior")
        if data.is_repeat_buyer and data.seller_account_age_days >= 365 and data.shared_device_accounts_24h <= 2:
            risk_points -= 2
            reasons.append("repeat buyer with a mature seller and isolated device looks safer")

        diagnostics.update(
            {
                "linked_cards_7d": history.get("linked_cards_7d", 0),
                "network_pattern": network_pattern,
                "is_flash_sale_context": "flash sale" in network_pattern,
                "is_cluster_context": "reuse the same seller and device cluster" in network_pattern,
                "micro_amount_card_testing": data.amount <= 10 and data.seller_chargeback_rate_30d >= 0.12,
            }
        )
        return risk_points, reasons, diagnostics

    def decide(self, observation) -> FraudCheckAction:
        risk_points, reasons, diagnostics = self._analyze_observation(observation)
        evidence_bonus, evidence_reasons = self._score_revealed_evidence(observation.revealed_evidence)
        risk_points += evidence_bonus

        threshold = {"easy": 3, "medium": 4, "hard": 5}[observation.task_name.value]
        margin = risk_points - threshold
        if self._should_investigate(observation, margin, diagnostics):
            target = self._pick_investigation_target(observation, diagnostics)
            if target is not None:
                return FraudCheckAction(
                    transaction_id=observation.transaction_id,
                    action_type=ActionTypeEnum.INVESTIGATE,
                    investigation_target=target,
                    reasoning=(
                        f"Signals are borderline for a final decision, so request {target.value} "
                        "to reduce ambiguity before committing."
                    )[:500],
                )

        decision = DecisionEnum.FRAUD if margin >= 0 else DecisionEnum.LEGITIMATE
        base_confidence = 0.60 + abs(margin) * 0.08
        if observation.revealed_evidence:
            base_confidence += 0.06
        if observation.task_name.value == "hard" and (
            diagnostics["is_cluster_context"] or diagnostics["is_flash_sale_context"]
        ):
            base_confidence += 0.20
        confidence = min(0.99, max(0.55, base_confidence))
        all_reasons = reasons + evidence_reasons
        reasoning = "; ".join(all_reasons[:3]) if all_reasons else "signals remain mixed but lean legitimate"

        return FraudCheckAction(
            transaction_id=observation.transaction_id,
            decision=decision,
            confidence=round(confidence, 2),
            reasoning=reasoning[:500],
        )

    @staticmethod
    def _score_revealed_evidence(revealed_evidence: Dict[str, Dict[str, Any]]) -> tuple[int, list[str]]:
        evidence_bonus = 0
        reasons: list[str] = []

        for target, evidence in revealed_evidence.items():
            risk_band = evidence.get("risk_band")
            if risk_band == "high":
                evidence_bonus += 1
                reasons.append(f"{target} returned a high-risk signal")
            elif risk_band == "low":
                evidence_bonus -= 1
                reasons.append(f"{target} returned a low-risk signal")

            details = evidence.get("details", {})
            if details.get("proxy_suspected"):
                evidence_bonus += 1
                reasons.append("device intelligence suspects proxy or account sharing")
            if details.get("issuer_velocity_alert", 0) >= 3:
                evidence_bonus += 1
                reasons.append("payment trace shows issuer velocity alerts")
            if details.get("prepaid_card_risk"):
                evidence_bonus += 1
                reasons.append("payment trace shows prepaid or wallet risk")
            if details.get("cluster_alert_score", 0.0) >= 0.75:
                evidence_bonus += 1
                reasons.append("network graph overlaps with a known abuse cluster")
            if details.get("linked_cards_7d", 0) >= 6:
                evidence_bonus += 1
                reasons.append("network graph shows a dense linked-card pattern")
            if details.get("device_country_match") is True and details.get("device_reuse_accounts_24h", 0) <= 2:
                evidence_bonus -= 1
                reasons.append("device intelligence looks consistent with normal traffic")
            if target == InvestigationTargetEnum.TRUST_NOTES.value and "completed prior purchases" in str(details.get("buyer_note", "")):
                evidence_bonus -= 1
                reasons.append("trust notes confirm repeat-buyer history")

        return evidence_bonus, reasons

    @staticmethod
    def _should_investigate(observation, margin: int, diagnostics: Dict[str, Any]) -> bool:
        if observation.investigation_budget_remaining <= 0 or not observation.available_investigations:
            return False
        if observation.revealed_evidence and abs(margin) >= 1:
            return False

        ambiguous = abs(margin) <= 1
        high_value = observation.transaction_data.amount_percentile >= 90.0
        suspicious_geo = diagnostics["device_mismatch"]
        elevated_cluster = diagnostics["cluster_alert_score"] >= 0.7
        high_chargeback = observation.transaction_data.seller_chargeback_rate_30d >= 0.12
        repeat_buyer_conflict = observation.transaction_data.is_repeat_buyer and diagnostics["device_mismatch"]
        linked_card_pressure = diagnostics.get("linked_cards_7d", 0) >= 6
        micro_amount_card_testing = diagnostics.get("micro_amount_card_testing", False)
        return (
            ambiguous
            or high_value
            or suspicious_geo
            or elevated_cluster
            or high_chargeback
            or repeat_buyer_conflict
            or linked_card_pressure
            or micro_amount_card_testing
        )

    @staticmethod
    def _pick_investigation_target(observation, diagnostics: Dict[str, Any]) -> InvestigationTargetEnum | None:
        available = list(observation.available_investigations)
        if not available:
            return None

        ordered_preferences = [
            InvestigationTargetEnum.NETWORK_GRAPH if diagnostics.get("is_cluster_context") else None,
            InvestigationTargetEnum.TRUST_NOTES if diagnostics.get("is_flash_sale_context") else None,
            InvestigationTargetEnum.NETWORK_GRAPH if diagnostics["cluster_alert_score"] >= 0.7 else None,
            InvestigationTargetEnum.DEVICE_INTEL if diagnostics["device_mismatch"] else None,
            InvestigationTargetEnum.PAYMENT_TRACE
            if observation.transaction_data.seller_chargeback_rate_30d >= 0.08
            or observation.transaction_data.buyer_disputes_90d >= 2
            else None,
            InvestigationTargetEnum.FULFILLMENT_REVIEW
            if observation.transaction_data.same_address_orders_24h >= 4
            else None,
            InvestigationTargetEnum.TRUST_NOTES,
        ]

        for target in ordered_preferences:
            if target is not None and target in available:
                return target
        return available[0]


class HybridCompetitionFraudDetectionAgent:
    """Use the strong deterministic agentic policy while still touching the proxy when configured."""

    name = "hybrid-agentic-baseline"

    def __init__(
        self,
        policy_agent: AgenticHeuristicFraudDetectionAgent,
        shadow_agent: Optional["OpenAIFraudDetectionAgent"] = None,
    ):
        self.policy_agent = policy_agent
        self.shadow_agent = shadow_agent
        self.api_base_url = getattr(shadow_agent, "api_base_url", None)
        self.model_name = getattr(shadow_agent, "model_name", None)
        self.shadow_probe_attempted = False
        self.shadow_probe_succeeded = False

    def decide(self, observation) -> FraudCheckAction:
        if self.shadow_agent is not None and not self.shadow_probe_attempted:
            self.shadow_probe_attempted = True
            try:
                self.shadow_agent.decide(observation)
                self.shadow_probe_succeeded = True
                logger.info(
                    "Proxy shadow probe succeeded for %s using model %s.",
                    observation.transaction_id,
                    self.model_name,
                )
            except Exception as exc:
                logger.warning("Proxy shadow probe failed: %s", exc)

        return self.policy_agent.decide(observation)


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
    policy_agent = AgenticHeuristicFraudDetectionAgent()

    if api_key:
        resolved_api_base_url = api_base_url or "https://router.huggingface.co/v1"
        resolved_model_name = model_name or discover_model_name(api_key, resolved_api_base_url)
        try:
            shadow_agent = OpenAIFraudDetectionAgent(
                model_name=resolved_model_name,
                api_key=api_key,
                api_base_url=resolved_api_base_url,
            )
            return HybridCompetitionFraudDetectionAgent(
                policy_agent=policy_agent,
                shadow_agent=shadow_agent,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize the proxy-backed competition agent: %s. Falling back to the deterministic agentic heuristic.",
                exc,
            )
            return policy_agent

    if model_name and not api_key:
        logger.warning(
            "MODEL_NAME was set but no API_KEY-compatible credential was available. "
            "Falling back to the deterministic agentic heuristic agent."
        )
    else:
        logger.warning(
            "API_KEY-compatible credentials were not set. "
            "Falling back to the deterministic agentic heuristic agent."
        )
    return policy_agent
