"""Baseline agents for the FraudShield FraudOps workflow."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum

try:  # pragma: no cover - optional in local smoke tests
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

logger = logging.getLogger(__name__)


def get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


class WorkflowHeuristicFraudOpsAgent:
    """Deterministic workflow baseline that follows a limited enterprise playbook."""

    name = "workflow-heuristic-baseline"

    def decide(self, observation) -> FraudCheckAction:
        case_id = observation.case_id
        revealed = observation.revealed_evidence
        task_name = observation.task_name.value
        case_summary = observation.case_summary

        if "transaction_review" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.REVIEW_TRANSACTION,
                reasoning="Open the case console before taking any further action.",
            )

        if task_name == "easy":
            if "merchant_profile" not in revealed and case_summary.visible_risk_band == "high":
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                    reasoning="A quick merchant check helps confirm the obvious queue signals.",
                )
            if case_summary.note_added is False:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.ADD_CASE_NOTE,
                    note_text="Reviewed the queue alert and merchant context before routing this obvious case.",
                )
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=self._resolve_easy(revealed),
                reasoning="The visible transaction and merchant signals are strong enough for final routing.",
            )

        if task_name == "medium":
            if "customer_profile" not in revealed:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
                    reasoning="Customer history is needed before making a policy-aware decision.",
                )
            if "policy_guide" not in revealed:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.CHECK_POLICY,
                    reasoning="Medium cases should be mapped against the policy guidance before routing.",
                )
            if case_summary.note_added is False:
                return FraudCheckAction(
                    case_id=case_id,
                    action_type=ActionTypeEnum.ADD_CASE_NOTE,
                    note_text="Reviewed customer history and policy triggers before selecting a medium-risk route.",
                )
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=self._resolve_medium(revealed),
                reasoning="The gathered profile and policy evidence are enough for a conservative medium-case route.",
            )

        if "network_graph" not in revealed and case_id.endswith("primary"):
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_NETWORK_GRAPH,
                reasoning="Primary hard cases need network evidence to understand linked abuse.",
            )
        if "merchant_profile" not in revealed and case_id.endswith("secondary"):
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                reasoning="Secondary hard cases get a merchant check before final routing.",
            )
        if "policy_guide" not in revealed and case_id.endswith("primary"):
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.CHECK_POLICY,
                reasoning="Escalation thresholds should be reviewed before closing the primary hard case.",
            )
        if case_summary.note_added is False:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.ADD_CASE_NOTE,
                note_text="Collected the available hard-case evidence and documented the current routing rationale.",
            )
        return FraudCheckAction(
            case_id=case_id,
            action_type=ActionTypeEnum.RESOLVE_CASE,
            resolution=self._resolve_hard(case_id, revealed),
            reasoning="The visible hard-case evidence is enough for a best-effort route, even if some linked detail remains hidden.",
        )

    def _resolve_easy(self, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        facts = revealed["transaction_review"]["facts"]
        geo_mismatch = facts["shipping_country"] != facts["device_country"]
        if facts["previous_fraud_flags"] >= 1 or facts["shared_device_accounts_24h"] >= 6 or geo_mismatch:
            return ResolutionEnum.BLOCK
        return ResolutionEnum.APPROVE

    def _resolve_medium(self, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        transaction_facts = revealed["transaction_review"]["facts"]
        customer_facts = revealed.get("customer_profile", {}).get("facts", {})
        if transaction_facts["previous_fraud_flags"] >= 1:
            return ResolutionEnum.HOLD
        if customer_facts.get("buyer_disputes_90d", 0) >= 3:
            return ResolutionEnum.HOLD
        if transaction_facts["previous_fraud_flags"] >= 1 or customer_facts.get("linked_cards_7d", 0) >= 5:
            return ResolutionEnum.REQUEST_DOCS
        return ResolutionEnum.APPROVE

    def _resolve_hard(self, case_id: str, revealed: Dict[str, Dict[str, Any]]) -> ResolutionEnum:
        transaction_facts = revealed["transaction_review"]["facts"]
        if case_id.endswith("primary"):
            cluster = revealed.get("network_graph", {}).get("facts", {}).get("cluster_alert_score", 0.0)
            return ResolutionEnum.ESCALATE if cluster >= 0.72 else ResolutionEnum.BLOCK
        if "network_graph" not in revealed:
            return ResolutionEnum.HOLD
        if transaction_facts["shared_device_accounts_24h"] >= 11:
            return ResolutionEnum.BLOCK
        return ResolutionEnum.HOLD


class RemoteLLMFraudOpsAgent:
    """OpenAI-compatible optional agent for the competition path."""

    name = "remote-llm-fraudops-agent"

    def __init__(self, model_name: str, api_key: str, api_base_url: Optional[str] = None, timeout: float = 30.0):
        if OpenAI is None:
            raise ImportError("openai package is not installed.")
        self.model_name = model_name
        self.api_base_url = api_base_url or "https://router.huggingface.co/v1"
        self.client = OpenAI(base_url=self.api_base_url, api_key=api_key, timeout=timeout)

    def decide(self, observation) -> FraudCheckAction:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                max_tokens=220,
                messages=self._build_messages(observation),
            )
            content = completion.choices[0].message.content or ""
            payload = self._parse_json(content)
            return FraudCheckAction.model_validate(payload)
        except Exception as exc:  # pragma: no cover - external API
            raise RuntimeError(
                "Remote FraudOps action generation failed. Check MODEL_NAME, API base URL, and credentials."
            ) from exc

    def _build_messages(self, observation) -> list[Dict[str, str]]:
        user_payload = {
            "case_id": observation.case_id,
            "task_name": observation.task_name.value,
            "current_screen": observation.current_screen.value,
            "visible_panels": observation.visible_panels,
            "revealed_evidence": observation.revealed_evidence,
            "linked_case_ids": observation.linked_case_ids,
            "remaining_steps": observation.remaining_steps,
            "remaining_sla": observation.remaining_sla,
            "note_required": observation.note_required,
            "allowed_actions": [action.value for action in observation.allowed_actions],
            "queue_items": [item.model_dump(mode="json") for item in observation.queue_items],
            "case_summary": observation.case_summary.model_dump(mode="json"),
            "app_context": observation.app_context,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an enterprise fraud operations analyst. Respond with JSON only. "
                    "Fields: case_id, action_type, note_text, resolution, reasoning. "
                    "Only use these action_type values: review_transaction, fetch_customer_profile, "
                    "fetch_merchant_profile, fetch_network_graph, check_policy, add_case_note, resolve_case. "
                    "When action_type is add_case_note, include note_text. "
                    "When action_type is resolve_case, include resolution and reasoning."
                ),
            },
            {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
        ]

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model did not return JSON.")
        return json.loads(text[start : end + 1])


def build_default_agent() -> object:
    """Build the configured remote agent, else return the deterministic workflow baseline."""

    model_name = get_env("MODEL_NAME", "MODELNAME")
    api_key = get_env("HF_TOKEN", "HFTOKEN", "OPENAI_API_KEY", "OPENAIAPIKEY", "API_KEY", "APIKEY")
    api_base_url = get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1")

    if model_name or api_key:
        if not model_name or not api_key:
            raise RuntimeError(
                "Both MODEL_NAME/MODELNAME and HF_TOKEN/HFTOKEN (or OPENAI_API_KEY/API_KEY) "
                "must be set for the remote LLM path."
            )
        return RemoteLLMFraudOpsAgent(model_name=model_name, api_key=api_key, api_base_url=api_base_url)

    logger.warning(
        "MODEL_NAME/MODELNAME and HF_TOKEN/HFTOKEN were not set. Falling back to the deterministic workflow baseline."
    )
    return WorkflowHeuristicFraudOpsAgent()
