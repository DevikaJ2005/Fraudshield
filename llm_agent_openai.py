"""LLM-backed agents for FraudShield."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum, TaskDifficulty

try:  # pragma: no cover - optional in local smoke tests
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

logger = logging.getLogger(__name__)

ACTION_ALIAS_TO_ENUM = {
    "merchant_profile": ActionTypeEnum.FETCH_MERCHANT_PROFILE,
    "fetch_merchant_profile": ActionTypeEnum.FETCH_MERCHANT_PROFILE,
    "customer_profile": ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
    "fetch_customer_profile": ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
    "network_graph": ActionTypeEnum.FETCH_NETWORK_GRAPH,
    "fetch_network_graph": ActionTypeEnum.FETCH_NETWORK_GRAPH,
    "device_intel": ActionTypeEnum.FETCH_NETWORK_GRAPH,
    "payment_trace": ActionTypeEnum.REVIEW_TRANSACTION,
    "fulfillment_review": ActionTypeEnum.REVIEW_TRANSACTION,
    "review_transaction": ActionTypeEnum.REVIEW_TRANSACTION,
    "policy_review": ActionTypeEnum.CHECK_POLICY,
    "check_policy": ActionTypeEnum.CHECK_POLICY,
    "trust_notes": ActionTypeEnum.CHECK_POLICY,
}

ACTION_ENUM_TO_ALIAS = {
    ActionTypeEnum.REVIEW_TRANSACTION: "payment_trace",
    ActionTypeEnum.FETCH_CUSTOMER_PROFILE: "customer_profile",
    ActionTypeEnum.FETCH_MERCHANT_PROFILE: "merchant_profile",
    ActionTypeEnum.FETCH_NETWORK_GRAPH: "network_graph",
    ActionTypeEnum.CHECK_POLICY: "policy_review",
}

ACTION_ENUM_TO_EVIDENCE_KEY = {
    ActionTypeEnum.REVIEW_TRANSACTION: "transaction_review",
    ActionTypeEnum.FETCH_CUSTOMER_PROFILE: "customer_profile",
    ActionTypeEnum.FETCH_MERCHANT_PROFILE: "merchant_profile",
    ActionTypeEnum.FETCH_NETWORK_GRAPH: "network_graph",
    ActionTypeEnum.CHECK_POLICY: "policy_guide",
}


class LLMFraudDetectionAgent:
    """OpenAI-compatible LLM agent with heuristic fallback."""

    name = "openai-compatible-llm-agent"
    agent_type = "llm_remote"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base_url: Optional[str] = None,
        fallback_agent: Optional[object] = None,
        timeout: float = 30.0,
    ):
        self.model_name = model_name or "gpt-4o-mini"
        self.api_base_url = api_base_url
        self.fallback_agent = fallback_agent
        self.timeout = timeout
        if OpenAI is None:
            self.client = None
        else:
            client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
            if api_base_url:
                client_kwargs["base_url"] = api_base_url
            self.client = OpenAI(**client_kwargs)

    def decide(self, observation) -> FraudCheckAction:
        if self.client is None:
            return self._fallback(observation, RuntimeError("openai package is not installed"))

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                max_tokens=260,
                messages=self._build_messages(observation),
            )
            content = completion.choices[0].message.content or ""
            payload = self._parse_json(content)
            return self._payload_to_action(payload, observation)
        except Exception as exc:  # pragma: no cover - external API
            return self._fallback(observation, exc)

    def _build_messages(self, observation) -> list[Dict[str, str]]:
        available_aliases = self._available_investigation_aliases(observation)
        observation_payload = {
            "case_id": observation.case_id,
            "task_name": observation.task_name.value,
            "current_screen": observation.current_screen.value,
            "visible_panels": observation.visible_panels,
            "case_summary": observation.case_summary.model_dump(mode="json"),
            "revealed_evidence": observation.revealed_evidence,
            "linked_case_ids": observation.linked_case_ids,
            "remaining_steps": observation.remaining_steps,
            "remaining_sla": observation.remaining_sla,
            "note_required": observation.note_required,
            "allowed_public_actions": [action.value for action in observation.allowed_actions],
            "available_investigation_aliases": available_aliases,
            "app_context": observation.app_context,
        }
        system_prompt = (
            "You are a fraud analyst operating inside a simulated investigation workflow. "
            "Only use the visible evidence shown to you. Choose either one investigation alias or one final "
            "decision. For investigation_target, you must return exactly one alias from "
            f"{available_aliases}. Never return placeholders, array expressions, or prose such as "
            "'available_investigations[0]'. Respond with JSON only using this schema: "
            '{"action_type":"investigate|decide","investigation_target":"string|null",'
            '"decision":"fraud|legitimate|null","confidence":0.0,"reasoning":"one sentence"}'
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(observation_payload, separators=(",", ":"))},
        ]

    def _payload_to_action(self, payload: Dict[str, Any], observation) -> FraudCheckAction:
        action_type = str(payload.get("action_type", "")).strip().lower()
        reasoning = self._normalize_reasoning(payload.get("reasoning"))
        if action_type == "investigate":
            investigation_target = str(payload.get("investigation_target", "")).strip().lower()
            mapped_action = self._map_investigation_alias(investigation_target, observation)
            mapped_action = self._stabilize_investigation_choice(mapped_action, observation)
            return FraudCheckAction(
                case_id=observation.case_id,
                action_type=mapped_action,
                reasoning=reasoning,
            )

        if action_type == "decide":
            decision = str(payload.get("decision", "")).strip().lower()
            confidence = self._coerce_confidence(payload.get("confidence"))
            if observation.note_required:
                note_text = self._build_note_text(reasoning)
                return FraudCheckAction(
                    case_id=observation.case_id,
                    action_type=ActionTypeEnum.ADD_CASE_NOTE,
                    note_text=note_text,
                )
            resolution = self._map_decision_to_resolution(decision, confidence, observation)
            return FraudCheckAction(
                case_id=observation.case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=resolution,
                reasoning=reasoning,
            )

        raise ValueError(f"Unsupported action_type from model: {action_type!r}")

    def _map_investigation_alias(self, alias: str, observation) -> ActionTypeEnum:
        normalized = alias.strip().lower()
        if normalized in ACTION_ALIAS_TO_ENUM:
            return ACTION_ALIAS_TO_ENUM[normalized]

        placeholder_match = re.fullmatch(r"available_investigations\[(\d+)\]", normalized)
        if placeholder_match:
            index = int(placeholder_match.group(1))
            available = self._available_investigation_aliases(observation)
            if 0 <= index < len(available):
                return ACTION_ALIAS_TO_ENUM[available[index]]

        compact = re.sub(r"[^a-z_]", "", normalized.replace("-", "_").replace(" ", "_"))
        for key, value in ACTION_ALIAS_TO_ENUM.items():
            if compact == key:
                return value
        for key, value in ACTION_ALIAS_TO_ENUM.items():
            if compact and compact in key:
                return value

        available = self._available_investigation_aliases(observation)
        if len(available) == 1:
            return ACTION_ALIAS_TO_ENUM[available[0]]
        raise ValueError(f"Unsupported investigation_target from model: {alias!r}")

    def _available_investigation_aliases(self, observation) -> list[str]:
        context_aliases = observation.app_context.get("available_investigations")
        aliases: list[str] = []
        if isinstance(context_aliases, list):
            for alias in context_aliases:
                normalized = str(alias).strip().lower()
                if normalized in ACTION_ALIAS_TO_ENUM:
                    canonical = ACTION_ENUM_TO_ALIAS[ACTION_ALIAS_TO_ENUM[normalized]]
                    if canonical not in aliases:
                        aliases.append(canonical)

        if aliases:
            return aliases

        fallback_aliases: list[str] = []
        for action in observation.allowed_actions:
            if action in ACTION_ENUM_TO_ALIAS:
                alias = ACTION_ENUM_TO_ALIAS[action]
                if alias not in fallback_aliases:
                    fallback_aliases.append(alias)
        return fallback_aliases

    def _stabilize_investigation_choice(self, action_type: ActionTypeEnum, observation) -> ActionTypeEnum:
        evidence_key = ACTION_ENUM_TO_EVIDENCE_KEY.get(action_type)
        if evidence_key and evidence_key not in observation.revealed_evidence:
            return action_type

        alternatives = []
        for alias in self._available_investigation_aliases(observation):
            candidate = ACTION_ALIAS_TO_ENUM[alias]
            candidate_key = ACTION_ENUM_TO_EVIDENCE_KEY.get(candidate)
            if candidate_key and candidate_key not in observation.revealed_evidence:
                alternatives.append(candidate)

        if alternatives:
            return alternatives[0]

        raise ValueError(
            f"Investigation {action_type.value!r} is already revealed and no unseen investigations remain."
        )

    def _map_decision_to_resolution(self, decision: str, confidence: float, observation) -> ResolutionEnum:
        if decision not in {"fraud", "legitimate"}:
            raise ValueError(f"Unsupported decision from model: {decision!r}")

        if decision == "legitimate":
            if confidence >= 0.75 or observation.task_name == TaskDifficulty.EASY:
                return ResolutionEnum.APPROVE
            return ResolutionEnum.REQUEST_DOCS

        network = observation.revealed_evidence.get("network_graph", {}).get("facts", {})
        if (
            observation.task_name == TaskDifficulty.HARD
            and observation.case_id.endswith("primary")
            and network.get("cluster_alert_score", 0.0) >= 0.75
            and network.get("linked_case_ids")
        ):
            return ResolutionEnum.ESCALATE
        if confidence < 0.70:
            return ResolutionEnum.HOLD
        return ResolutionEnum.BLOCK

    def _build_note_text(self, reasoning: str) -> str:
        note = f"Decision summary: {reasoning}"
        if len(note) < 12:
            note = "Decision summary: routing based on the currently visible investigation evidence."
        return note

    def _fallback(self, observation, exc: Exception) -> FraudCheckAction:
        logger.warning("LLM agent failed, falling back to heuristic: %s", exc)
        if self.fallback_agent is None:
            raise RuntimeError("No fallback agent configured for LLM failure.") from exc
        return self.fallback_agent.decide(observation)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model did not return JSON.")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _normalize_reasoning(value: Any) -> str:
        reasoning = str(value or "").strip()
        if len(reasoning) < 12:
            reasoning = "Routing based on the currently visible investigation evidence."
        return reasoning

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))


class LocalModelFraudDetectionAgent(LLMFraudDetectionAgent):
    """Local HF/PEFT model path for notebook-produced checkpoints."""

    name = "local-model-llm-agent"
    agent_type = "llm_local"

    def __init__(self, model_path: str, fallback_agent: Optional[object] = None):
        self.model_path = str(model_path)
        self.fallback_agent = fallback_agent
        self.model_name = Path(model_path).name
        self.api_base_url = None
        self.timeout = 0.0
        self.client = None
        self.model = None
        self.tokenizer = None
        self._load_model()

    def decide(self, observation) -> FraudCheckAction:
        try:
            prompt = self._build_local_prompt(observation)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            content = self.tokenizer.decode(generated, skip_special_tokens=True)
            payload = self._parse_json(content)
            return self._payload_to_action(payload, observation)
        except Exception as exc:
            return self._fallback(observation, exc)

    def _build_local_prompt(self, observation) -> str:
        messages = self._build_messages(observation)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - notebook/runtime dependency
            raise ImportError("Local model inference requires transformers and torch.") from exc

        model_dir = Path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            try:
                from peft import AutoPeftModelForCausalLM

                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype="auto",
                    device_map="auto",
                )
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Local adapter inference requires peft to be installed.") from exc
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype="auto",
                device_map="auto",
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, "eval"):
            self.model.eval()
