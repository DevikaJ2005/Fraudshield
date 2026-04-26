"""Reward decomposition helpers for RL-style FraudShield training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from models import ActionTypeEnum, ResolutionEnum


@dataclass
class RewardBreakdown:
    """Structured numeric reward with interpretable subscores."""

    env_reward: float
    correctness: float
    task_completion: float
    reasoning_quality: float
    efficiency: float
    safety: float
    formatting_compliance: float
    consistency: float
    total_reward: float

    def to_dict(self) -> dict[str, float]:
        return {
            "env_reward": self.env_reward,
            "correctness": self.correctness,
            "task_completion": self.task_completion,
            "reasoning_quality": self.reasoning_quality,
            "efficiency": self.efficiency,
            "safety": self.safety,
            "formatting_compliance": self.formatting_compliance,
            "consistency": self.consistency,
            "total_reward": self.total_reward,
        }


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_reasoning_quality(reasoning: str, revealed_evidence: dict[str, Any]) -> float:
    """Reward concise evidence-aware reasoning."""

    reasoning = (reasoning or "").strip().lower()
    if len(reasoning) < 12:
        return -0.4
    signal_hits = 0
    for evidence_key in revealed_evidence:
        stem = evidence_key.replace("_", " ")
        if any(token in reasoning for token in stem.split()):
            signal_hits += 1
    return _clamp(0.2 + 0.2 * signal_hits, -1.0, 1.0)


def score_efficiency(remaining_steps: int, initial_budget: int, token_count: int) -> float:
    """Reward shorter trajectories and lower token usage."""

    if initial_budget <= 0:
        return 0.0
    step_ratio = remaining_steps / initial_budget
    token_penalty = min(token_count / 300.0, 1.0)
    return _clamp((step_ratio * 0.8) - (token_penalty * 0.4))


def score_safety(action_type: ActionTypeEnum, parse_failed: bool, refused_unsafely: bool = False) -> float:
    """Reward well-formed safe handling."""

    if parse_failed:
        return -1.0
    if refused_unsafely:
        return -0.7
    if action_type == ActionTypeEnum.RESOLVE_CASE:
        return 0.3
    return 0.5


def score_formatting_compliance(parse_failed: bool, required_fields_present: bool) -> float:
    """Reward JSON compliance and field completeness."""

    if parse_failed:
        return -1.0
    return 1.0 if required_fields_present else -0.4


def score_consistency(action_history: Iterable[str], next_action: str, resolution: ResolutionEnum | None) -> float:
    """Reward non-redundant consistent behavior."""

    history = list(action_history)
    if history and history[-1] == next_action and next_action.startswith("fetch_"):
        return -0.8
    if resolution is not None and history.count("resolve_case") > 0:
        return -1.0
    return 0.4


def score_correctness(env_reward_value: float, is_correct: bool | None) -> float:
    """Expose final correctness separately from raw environment reward."""

    if is_correct is True:
        return 1.0
    if is_correct is False:
        return -1.0
    return _clamp(env_reward_value)


def score_task_completion(done: bool, action_type: ActionTypeEnum, resolution: ResolutionEnum | None) -> float:
    """Reward finishing the case and using the right action family."""

    if done and action_type == ActionTypeEnum.RESOLVE_CASE and resolution is not None:
        return 1.0
    if action_type == ActionTypeEnum.ADD_CASE_NOTE:
        return 0.3
    return 0.1 if done else 0.0


def build_reward_breakdown(
    *,
    env_reward_value: float,
    is_correct: bool | None,
    done: bool,
    action_type: ActionTypeEnum,
    resolution: ResolutionEnum | None,
    reasoning: str,
    revealed_evidence: dict[str, Any],
    remaining_steps: int,
    initial_budget: int,
    token_count: int,
    parse_failed: bool,
    required_fields_present: bool,
    action_history: Iterable[str],
    weights: Any,
) -> RewardBreakdown:
    """Build a decomposed scalar reward for RL loops."""

    correctness = score_correctness(env_reward_value, is_correct)
    task_completion = score_task_completion(done, action_type, resolution)
    reasoning_quality = score_reasoning_quality(reasoning, revealed_evidence)
    efficiency = score_efficiency(remaining_steps, initial_budget, token_count)
    safety = score_safety(action_type, parse_failed=parse_failed)
    formatting = score_formatting_compliance(parse_failed=parse_failed, required_fields_present=required_fields_present)
    consistency = score_consistency(action_history, action_type.value, resolution)

    total_reward = (
        weights.env_reward * env_reward_value
        + weights.correctness * correctness
        + weights.task_completion * task_completion
        + weights.reasoning_quality * reasoning_quality
        + weights.efficiency * efficiency
        + weights.safety * safety
        + weights.formatting_compliance * formatting
        + weights.consistency * consistency
    )

    return RewardBreakdown(
        env_reward=env_reward_value,
        correctness=correctness,
        task_completion=task_completion,
        reasoning_quality=reasoning_quality,
        efficiency=efficiency,
        safety=safety,
        formatting_compliance=formatting,
        consistency=consistency,
        total_reward=total_reward,
    )
