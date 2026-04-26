"""Text-first training environment wrapper for FraudShield."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config import EnvironmentConfig, RewardWeights
from fraudshield_env import FraudShieldEnvironment
from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum
from reward import RewardBreakdown, build_reward_breakdown
from utils import approximate_token_count, extract_json_object

CANONICAL_INVESTIGATION_ALIASES = [
    "merchant_profile",
    "customer_profile",
    "network_graph",
    "payment_trace",
    "policy_review",
]

INVESTIGATION_ALIAS_TO_ACTION = {
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
}

ACTION_TYPE_TO_CANONICAL_ALIAS = {
    ActionTypeEnum.FETCH_MERCHANT_PROFILE: "merchant_profile",
    ActionTypeEnum.FETCH_CUSTOMER_PROFILE: "customer_profile",
    ActionTypeEnum.FETCH_NETWORK_GRAPH: "network_graph",
    ActionTypeEnum.REVIEW_TRANSACTION: "payment_trace",
    ActionTypeEnum.CHECK_POLICY: "policy_review",
}


def build_fraudshield_prompt(observation) -> str:
    """Build the canonical prompt used for both training and inference."""

    payload = {
        "case_id": observation.case_id,
        "task_name": observation.task_name.value,
        "visible_panels": observation.visible_panels,
        "revealed_evidence": observation.revealed_evidence,
        "linked_case_ids": observation.linked_case_ids,
        "remaining_steps": observation.remaining_steps,
        "remaining_sla": observation.remaining_sla,
        "note_required": observation.note_required,
        "allowed_actions": [action.value for action in observation.allowed_actions],
        "case_summary": observation.case_summary.model_dump(mode="json"),
        "app_context": observation.app_context,
    }
    available = observation.app_context.get("available_investigations", CANONICAL_INVESTIGATION_ALIASES)
    return (
        "You are a fraud analyst in a multi-step training environment. "
        "Return JSON only. Use visible evidence, investigation budget, and prior evidence carefully.\n\n"
        f"Visible observation:\n{json.dumps(payload, sort_keys=True)}\n\n"
        f"Valid investigation aliases: {available}\n"
        "JSON schema: "
        '{"action_type":"investigate|decide","investigation_target":"alias_or_null",'
        '"decision":"fraud|legitimate|null","confidence":0.0,"reasoning":"one sentence"}'
    )


@dataclass
class TextStepResult:
    """Structured step output for text-based RL loops."""

    prompt: str
    response_text: str
    next_prompt: str
    done: bool
    reward: float
    reward_breakdown: RewardBreakdown
    info: dict[str, Any]


class FraudShieldTextEnvironment:
    """Wrap ``FraudShieldEnvironment`` as a text-in/text-out RL environment."""

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        reward_weights: RewardWeights | None = None,
    ):
        self.env_config = env_config or EnvironmentConfig()
        self.reward_weights = reward_weights or RewardWeights()
        self.env = FraudShieldEnvironment(data_path=self.env_config.data_path, seed=self.env_config.seed)
        self.env.load_data()
        self.current_observation = None
        self.current_task = self.env_config.default_task
        self.initial_step_budget = self.env_config.max_rollout_steps
        self.action_history: list[str] = []

    def reset(self, task: str | None = None) -> str:
        """Reset the wrapped environment and return the initial prompt."""

        self.current_task = task or self.current_task
        result = self.env.reset(task=self.current_task)
        self.current_observation = result.observation
        self.initial_step_budget = result.info.get("max_steps", self.env_config.max_rollout_steps)
        self.action_history = []
        return self.build_prompt(self.current_observation)

    def build_prompt(self, observation) -> str:
        """Build the prompt shown to an LLM policy."""
        return build_fraudshield_prompt(observation)

    def parse_response(self, response_text: str) -> tuple[FraudCheckAction, dict[str, Any], bool, bool]:
        """Convert model output into a typed environment action."""

        parse_failed = False
        required_fields_present = True
        try:
            payload = extract_json_object(response_text)
        except Exception:
            parse_failed = True
            required_fields_present = False
            payload = {
                "action_type": "investigate",
                "investigation_target": "payment_trace",
                "decision": None,
                "confidence": 0.0,
                "reasoning": "Fallback after invalid output.",
            }

        action_type = str(payload.get("action_type", "")).strip().lower()
        reasoning = str(payload.get("reasoning", "")).strip()
        if not reasoning:
            required_fields_present = False
            reasoning = "Fallback after missing reasoning."

        if action_type == "investigate":
            alias = str(payload.get("investigation_target", "")).strip().lower()
            if not alias:
                required_fields_present = False
                alias = "payment_trace"
            mapped_action = INVESTIGATION_ALIAS_TO_ACTION.get(alias, ActionTypeEnum.REVIEW_TRANSACTION)
            action = FraudCheckAction(case_id=self.current_observation.case_id, action_type=mapped_action, reasoning=reasoning)
        elif action_type == "decide":
            decision = str(payload.get("decision", "")).strip().lower()
            confidence = float(payload.get("confidence") or 0.5)
            if decision not in {"fraud", "legitimate"}:
                required_fields_present = False
                decision = "fraud"
            if self.current_observation.note_required:
                action = FraudCheckAction(
                    case_id=self.current_observation.case_id,
                    action_type=ActionTypeEnum.ADD_CASE_NOTE,
                    note_text=f"Decision summary: {reasoning}",
                )
            else:
                resolution = self._decision_to_resolution(decision, confidence)
                action = FraudCheckAction(
                    case_id=self.current_observation.case_id,
                    action_type=ActionTypeEnum.RESOLVE_CASE,
                    resolution=resolution,
                    reasoning=reasoning,
                )
        else:
            required_fields_present = False
            action = FraudCheckAction(
                case_id=self.current_observation.case_id,
                action_type=ActionTypeEnum.REVIEW_TRANSACTION,
                reasoning="Fallback after unsupported action type.",
            )
        return action, payload, parse_failed, required_fields_present

    def step(self, response_text: str) -> TextStepResult:
        """Step the environment using raw model text."""

        prompt = self.build_prompt(self.current_observation)
        action, payload, parse_failed, required_fields_present = self.parse_response(response_text)
        env_step = self.env.step(action)
        self.action_history.append(action.action_type.value)
        self.current_observation = env_step.observation
        token_count = approximate_token_count(prompt + response_text)
        breakdown = build_reward_breakdown(
            env_reward_value=env_step.reward.value,
            is_correct=env_step.reward.is_correct,
            done=env_step.done,
            action_type=action.action_type,
            resolution=action.resolution,
            reasoning=action.reasoning if action.action_type != ActionTypeEnum.ADD_CASE_NOTE else action.note_text or "",
            revealed_evidence=env_step.observation.revealed_evidence,
            remaining_steps=env_step.observation.remaining_steps,
            initial_budget=self.initial_step_budget,
            token_count=token_count,
            parse_failed=parse_failed,
            required_fields_present=required_fields_present,
            action_history=self.action_history[:-1],
            weights=self.reward_weights,
        )
        next_prompt = self.build_prompt(self.current_observation)
        return TextStepResult(
            prompt=prompt,
            response_text=response_text,
            next_prompt=next_prompt,
            done=env_step.done,
            reward=breakdown.total_reward,
            reward_breakdown=breakdown,
            info={
                "payload": payload,
                "env_reward": env_step.reward.model_dump(mode="json"),
                "state": self.env.state().model_dump(mode="json"),
            },
        )

    def _decision_to_resolution(self, decision: str, confidence: float) -> ResolutionEnum:
        if decision == "legitimate":
            if confidence >= 0.75 or self.current_observation.task_name.value == "easy":
                return ResolutionEnum.APPROVE
            return ResolutionEnum.REQUEST_DOCS
        if confidence < 0.70:
            return ResolutionEnum.HOLD
        return ResolutionEnum.BLOCK
