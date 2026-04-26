#!/usr/bin/env python3
"""Competition inference for the FraudShield investigation environment."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Tuple

from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import SnapshotCalibratedFraudDetectionAgent, build_default_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "fraudshield_baseline_results.json"


def get_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


def emit_event(event_name: str, **fields: object) -> None:
    """Print validator-friendly structured progress blocks to stdout."""

    parts = [f"[{event_name}]"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def build_resilient_agent() -> Tuple[object, object]:
    """Prefer the configured agent but keep a clean heuristic fallback."""

    heuristic = SnapshotCalibratedFraudDetectionAgent()
    try:
        return build_default_agent(), heuristic
    except Exception as exc:
        logger.warning("Agent initialization failed: %s. Falling back to heuristic baseline.", exc)
        return heuristic, heuristic


def run_task(
    env: FraudShieldEnvironment,
    agent: object,
    fallback_agent: SnapshotCalibratedFraudDetectionAgent,
    task_name: str,
) -> Tuple[Dict[str, object], object, List[Dict[str, object]], List[Dict[str, object]], bool]:
    """Run a full workflow episode for one task."""

    configured_agent = agent
    agent_name = getattr(agent, "name", agent.__class__.__name__)
    emit_event("START", task=task_name, agent=agent_name)
    logger.info("START %s %s", task_name.upper(), agent_name)

    observation = env.reset(task_name).observation
    action_trace: List[Dict[str, object]] = []
    final_decisions: List[Dict[str, object]] = []
    fallback_triggered = False

    while not env.is_done:
        try:
            action = agent.decide(observation)
        except Exception as exc:
            fallback_triggered = True
            logger.warning(
                "Agent decision failed on task %s at step %s: %s. Switching to heuristic fallback.",
                task_name,
                env.step_count + 1,
                exc,
            )
            agent = fallback_agent
            action = agent.decide(observation)

        step_result = env.step(action)
        trace_event = {
            "step": env.step_count,
            "case_id": action.case_id,
            "action_type": action.action_type.value,
            "reasoning": action.reasoning,
            "reward": step_result.reward.value,
            "done": step_result.done,
        }
        if action.note_text:
            trace_event["note_text"] = action.note_text
        if action.resolution is not None:
            trace_event["resolution"] = action.resolution.value
            final_decisions.append(
                {
                    "step": env.step_count,
                    "case_id": action.case_id,
                    "resolution": action.resolution.value,
                    "reasoning": action.reasoning,
                    "reward": step_result.reward.value,
                }
            )
        action_trace.append(trace_event)

        emit_fields = {
            "task": task_name,
            "step": env.step_count,
            "action": action.action_type.value,
            "case_id": action.case_id,
            "reward": f"{step_result.reward.value:+.2f}",
        }
        if action.resolution is not None:
            emit_fields["resolution"] = action.resolution.value
        emit_event("STEP", **emit_fields)
        logger.info(
            "STEP %02d %s %s %+.2f",
            env.step_count,
            action.action_type.value,
            action.case_id,
            step_result.reward.value,
        )
        observation = step_result.observation

    summary = env.get_episode_report()
    emit_event(
        "END",
        task=task_name,
        steps=summary["step_count"],
        reward=f"{summary['cumulative_reward']:+.3f}",
        accuracy=f"{summary['metrics']['resolution_accuracy']:.3f}",
    )
    logger.info(
        "END %s accuracy=%.3f reward=%.3f",
        task_name.upper(),
        summary["metrics"]["resolution_accuracy"],
        summary["cumulative_reward"],
    )
    summary["configured_agent_name"] = getattr(configured_agent, "name", configured_agent.__class__.__name__)
    summary["effective_agent_name"] = getattr(agent, "name", agent.__class__.__name__)
    return summary, agent, action_trace, final_decisions, fallback_triggered


def main() -> Dict[str, object]:
    """Run the configured agent across easy, medium, and hard tasks."""

    logger.info("%s", "=" * 72)
    logger.info("FraudShield baseline inference")
    logger.info("%s", "=" * 72)

    env = FraudShieldEnvironment(data_path="data", seed=42)
    if not env.load_data():
        logger.error("FraudShield data could not be loaded from ./data")
        sys.exit(1)

    agent, fallback_agent = build_resilient_agent()
    configured_agent_name = getattr(agent, "name", agent.__class__.__name__)
    configured_agent_type = getattr(agent, "agent_type", "unknown")
    logger.info(
        "Configured agent: %s (%s) | API_BASE_URL=%s | MODEL_NAME=%s | LOCAL_MODEL_PATH=%s | HF_TOKEN=%s",
        configured_agent_name,
        configured_agent_type,
        get_env("API_BASE_URL", default="<default>"),
        get_env("MODEL_NAME", default="<unset>"),
        get_env("LOCAL_MODEL_PATH", default="<unset>"),
        "<set>" if get_env("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN") else "<unset>",
    )

    easy_summary, agent, easy_trace, easy_decisions, easy_fallback = run_task(env, agent, fallback_agent, "easy")
    medium_summary, agent, medium_trace, medium_decisions, medium_fallback = run_task(
        env, agent, fallback_agent, "medium"
    )
    hard_summary, agent, hard_trace, hard_decisions, hard_fallback = run_task(env, agent, fallback_agent, "hard")

    grading_result = FraudShieldGrader.grade_all_tasks(easy_summary, medium_summary, hard_summary)
    grading_result["metadata"] = {
        "configured_agent_name": configured_agent_name,
        "configured_agent_type": configured_agent_type,
        "effective_agent_name": getattr(agent, "name", agent.__class__.__name__),
        "effective_agent_type": getattr(agent, "agent_type", "unknown"),
        "fallback_triggered": easy_fallback or medium_fallback or hard_fallback,
        "api_base_url": get_env("API_BASE_URL"),
        "model_name": get_env("MODEL_NAME", default="gpt-4o-mini"),
        "local_model_path": get_env("LOCAL_MODEL_PATH"),
        "hf_token_present": bool(get_env("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")),
        "seed": 42,
        "data_snapshot": env.data_loader.get_bundle_summary(),
        "task_steps": {
            "easy": easy_summary["step_count"],
            "medium": medium_summary["step_count"],
            "hard": hard_summary["step_count"],
        },
    }
    grading_result["episode_summaries"] = {
        "easy": easy_summary,
        "medium": medium_summary,
        "hard": hard_summary,
    }
    grading_result["action_traces"] = {
        "easy": easy_trace,
        "medium": medium_trace,
        "hard": hard_trace,
    }
    grading_result["final_decisions"] = {
        "easy": easy_decisions,
        "medium": medium_decisions,
        "hard": hard_decisions,
    }

    logger.info("Easy score:   %.4f", grading_result["easy"]["score"])
    logger.info("Medium score: %.4f", grading_result["medium"]["score"])
    logger.info("Hard score:   %.4f", grading_result["hard"]["score"])
    logger.info("Final score:  %.4f", grading_result["final_score"])

    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(grading_result, handle, indent=2)
    logger.info("Saved baseline report to %s", RESULTS_FILE)
    return grading_result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Baseline inference failed: %s", exc)
        sys.exit(1)
