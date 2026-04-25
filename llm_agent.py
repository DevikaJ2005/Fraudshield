"""Agent selection and heuristic baseline for FraudShield."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum

logger = logging.getLogger(__name__)

HIGH_VALUE_CATEGORIES = {"luxury", "electronics", "travel", "high_value_collectibles", "collectibles"}
RISKY_PAYMENT_METHODS = {"prepaid_card", "gift_card", "crypto_gateway"}


def get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


class SnapshotCalibratedFraudDetectionAgent:
    """Deterministic baseline tuned for the hidden-evidence workflow."""

    name = "snapshot-calibrated-heuristic"
    agent_type = "heuristic"

    def decide(self, observation) -> FraudCheckAction:
        case_id = observation.case_id
        revealed = observation.revealed_evidence
        task_name = observation.task_name.value
        budget = int(observation.app_context.get("investigation_budget_remaining", 0))
        item_category = str(observation.app_context.get("item_category", ""))
        amount = float(observation.case_summary.amount_usd)

        if "transaction_review" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.REVIEW_TRANSACTION,
                reasoning="Open the transaction trace before any deeper investigation.",
            )

        if task_name == "easy":
            if budget > 0 and "merchant_profile" not in revealed and (
                amount >= 200.0 or item_category in HIGH_VALUE_CATEGORIES
            ):
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                    reasoning="A single merchant review is enough to confirm this easy case.",
                )
            if observation.note_required:
                return self._note_action(
                    case_id,
                    "Reviewed the transaction trace and captured the visible merchant risk before routing.",
                )
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=self._resolve_easy(revealed),
                reasoning="The visible transaction pattern is sufficient for an easy-case route.",
            )

        if task_name == "medium":
            if budget > 0 and "customer_profile" not in revealed:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
                    reasoning="Customer context is needed before deciding this mixed-signal case.",
                )
            if budget > 0 and "policy_guide" not in revealed:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.CHECK_POLICY,
                    reasoning="Policy guidance helps separate a hold from a document request.",
                )
            if budget > 0 and "merchant_profile" not in revealed and self._transaction_looks_risky(revealed):
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                    reasoning="Merchant context can resolve the remaining ambiguity in this medium case.",
                )
            if observation.note_required:
                return self._note_action(
                    case_id,
                    "Reviewed the available customer, transaction, and policy evidence before routing the case.",
                )
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=self._resolve_medium(revealed),
                reasoning="The combined medium-case evidence supports a conservative final route.",
            )

        if budget > 0 and "network_graph" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_NETWORK_GRAPH,
                reasoning="Hard cases usually need graph evidence before the routing becomes reliable.",
            )
        if case_id.endswith("primary") and budget > 0 and "merchant_profile" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                reasoning="Merchant risk helps determine whether the primary hard case should escalate.",
            )
        if case_id.endswith("secondary") and budget > 0 and "customer_profile" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
                reasoning="Customer context helps determine whether the secondary hard case should block or hold.",
            )
        if budget > 0 and "policy_guide" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.CHECK_POLICY,
                reasoning="Policy is needed before choosing the final route on a hard case.",
            )
        if observation.note_required:
            return self._note_action(
                case_id,
                "Captured the reviewed transaction, graph, and supporting evidence before closing the hard case.",
            )
        return FraudCheckAction(
            case_id=case_id,
            action_type=ActionTypeEnum.RESOLVE_CASE,
            resolution=self._resolve_hard(case_id, revealed),
            reasoning="The available hard-case evidence supports the strongest route currently justified.",
        )

    def _note_action(self, case_id: str, note_text: str) -> FraudCheckAction:
        return FraudCheckAction(
            case_id=case_id,
            action_type=ActionTypeEnum.ADD_CASE_NOTE,
            note_text=note_text,
        )

    def _transaction_looks_risky(self, revealed: Dict[str, Dict[str, Any]]) -> bool:
        facts = revealed.get("transaction_review", {}).get("facts", {})
        if not facts:
            return False
        return bool(
            facts.get("payment_method") in RISKY_PAYMENT_METHODS
            or facts.get("same_address_orders_24h", 0) >= 4
            or facts.get("device_country") != facts.get("shipping_country")
            or facts.get("shipping_speed") in {"overnight", "same-day"}
        )

    def _resolve_easy(self, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        facts = revealed["transaction_review"]["facts"]
        merchant = revealed.get("merchant_profile", {}).get("facts", {})
        if (
            facts.get("payment_method") in RISKY_PAYMENT_METHODS
            or facts.get("same_address_orders_24h", 0) >= 4
            or facts.get("device_country") != facts.get("shipping_country")
            or merchant.get("seller_chargeback_rate_30d", 0.0) >= 0.10
            or merchant.get("seller_account_age_days", 9999) <= 45
        ):
            return ResolutionEnum.BLOCK
        return ResolutionEnum.APPROVE

    def _resolve_medium(self, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        facts = revealed["transaction_review"]["facts"]
        customer = revealed.get("customer_profile", {}).get("facts", {})
        merchant = revealed.get("merchant_profile", {}).get("facts", {})
        if not merchant and customer.get("buyer_disputes_90d", 0) >= 2:
            return ResolutionEnum.HOLD
        conflict_score = 0
        if customer.get("buyer_disputes_90d", 0) >= 2:
            conflict_score += 1
        if not customer.get("is_repeat_buyer", True):
            conflict_score += 1
        if merchant.get("seller_chargeback_rate_30d", 0.0) >= 0.06:
            conflict_score += 1
        if facts.get("payment_method") in RISKY_PAYMENT_METHODS:
            conflict_score += 1
        if conflict_score >= 3:
            return ResolutionEnum.HOLD
        if conflict_score >= 1:
            return ResolutionEnum.REQUEST_DOCS
        return ResolutionEnum.APPROVE

    def _resolve_hard(self, case_id: str, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        network = revealed.get("network_graph", {}).get("facts", {})
        merchant = revealed.get("merchant_profile", {}).get("facts", {})
        customer = revealed.get("customer_profile", {}).get("facts", {})

        if case_id.endswith("primary"):
            if network.get("cluster_alert_score", 0.0) >= 0.75 and network.get("linked_case_ids"):
                return ResolutionEnum.ESCALATE
            if merchant.get("seller_chargeback_rate_30d", 0.0) >= 0.10:
                return ResolutionEnum.BLOCK
            return ResolutionEnum.HOLD

        if network.get("shared_device_accounts_24h", 0) >= 6 or network.get("previous_fraud_flags", 0) >= 1:
            return ResolutionEnum.BLOCK
        if customer.get("buyer_disputes_90d", 0) >= 2:
            return ResolutionEnum.HOLD
        return ResolutionEnum.HOLD


def build_default_agent() -> object:
    """Build the best available agent for the current runtime."""

    heuristic = SnapshotCalibratedFraudDetectionAgent()
    local_model_path = get_env("LOCAL_MODEL_PATH")
    model_name = get_env("MODEL_NAME", default="gpt-4o-mini")
    api_key = get_env("API_KEY", "OPENAI_API_KEY")
    api_base_url = get_env("API_BASE_URL")

    if local_model_path:
        from llm_agent_openai import LocalModelFraudDetectionAgent

        return LocalModelFraudDetectionAgent(
            model_path=local_model_path,
            fallback_agent=heuristic,
        )

    if api_key:
        from llm_agent_openai import LLMFraudDetectionAgent

        return LLMFraudDetectionAgent(
            model_name=model_name or "gpt-4o-mini",
            api_key=api_key,
            api_base_url=api_base_url,
            fallback_agent=heuristic,
        )

    logger.warning("No LOCAL_MODEL_PATH or API_KEY found. Falling back to the calibrated heuristic baseline.")
    return heuristic
