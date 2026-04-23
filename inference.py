#!/usr/bin/env python3
"""Competition baseline inference for the FraudShield FraudOps workflow."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, Tuple

from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import WorkflowHeuristicFraudOpsAgent, build_default_agent

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


def build_resilient_agent() -> object:
    """Prefer the configured agent but fall back to the deterministic workflow baseline."""

    try:
        return build_default_agent()
    except Exception as exc:
        logger.warning("Agent initialization failed: %s. Falling back to workflow heuristic baseline.", exc)
        return WorkflowHeuristicFraudOpsAgent()


def run_task(env: FraudShieldEnvironment, agent: object, task_name: str) -> Tuple[Dict[str, object], object]:
    """Run a full workflow episode for one task."""

    agent_name = getattr(agent, "name", agent.__class__.__name__)
    logger.info("START %s %s", task_name.upper(), agent_name)
    emit_event("START", task=task_name, agent=agent_name)

    observation = env.reset(task_name).observation
    fallback_agent: object | None = None

    while not env.is_done:
        try:
            action = agent.decide(observation)
        except Exception as exc:
            if fallback_agent is None:
                fallback_agent = WorkflowHeuristicFraudOpsAgent()
                logger.warning(
                    "Agent decision failed on task %s at step %s: %s. Switching to workflow heuristic fallback.",
                    task_name,
                    env.step_count + 1,
                    exc,
                )
            agent = fallback_agent
            action = agent.decide(observation)

        step_result = env.step(action)
        emit_fields = {
            "task": task_name,
            "step": env.step_count,
            "action": action.action_type.value,
            "case_id": action.case_id,
            "reward": f"{step_result.reward.value:+.2f}",
        }
        if action.note_text:
            emit_fields["note"] = "yes"
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
    return summary, agent


def main() -> Dict[str, object]:
    """Run the baseline across easy/medium/hard and save the final report."""

    logger.info("%s", "=" * 72)
    logger.info("FraudShield FraudOps baseline inference")
    logger.info("%s", "=" * 72)

    env = FraudShieldEnvironment(data_path="data", seed=42)
    if not env.load_data():
        logger.error("FraudShield data could not be loaded from ./data")
        sys.exit(1)

    agent = build_resilient_agent()
    logger.info(
        "Agent mode: %s | API_BASE_URL=%s | MODEL_NAME=%s",
        getattr(agent, "name", agent.__class__.__name__),
        get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1"),
        get_env("MODEL_NAME", "MODELNAME", default="<offline-workflow-heuristic>"),
    )

    easy_summary, agent = run_task(env, agent, "easy")
    medium_summary, agent = run_task(env, agent, "medium")
    hard_summary, agent = run_task(env, agent, "hard")

    grading_result = FraudShieldGrader.grade_all_tasks(easy_summary, medium_summary, hard_summary)
    grading_result["metadata"] = {
        "agent_name": getattr(agent, "name", agent.__class__.__name__),
        "api_base_url": get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1"),
        "model_name": get_env("MODEL_NAME", "MODELNAME"),
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
