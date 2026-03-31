#!/usr/bin/env python3
"""Competition baseline inference for FraudShield."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Tuple

from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import build_default_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "fraudshield_baseline_results.json"


def get_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def run_task(env: FraudShieldEnvironment, agent: object, task_name: str) -> Tuple[List[str], List[str], List[float]]:
    """Run one task episode and capture the full prediction trace."""

    logger.info("%s", "=" * 72)
    logger.info("Running %s task with %s", task_name.upper(), getattr(agent, "name", agent.__class__.__name__))
    logger.info("%s", "=" * 72)

    reset_result = env.reset(task_name)
    logger.info("Episode %s contains %s transactions", env.episode_id, reset_result.info["num_transactions"])

    observation = reset_result.observation
    predictions: List[str] = []
    confidences: List[float] = []

    while not env.is_done:
        action = agent.decide(observation)
        predictions.append(action.decision.value)
        confidences.append(action.confidence)
        step_result = env.step(action)

        if env.step_count in {1, len(env.current_cases)} or env.step_count % 10 == 0:
            logger.info(
                "Step %02d | decision=%s | confidence=%.2f | reward=%+.2f",
                env.step_count,
                action.decision.value,
                action.confidence,
                step_result.reward.value,
            )

        observation = step_result.observation

    logger.info(
        "Finished %s: accuracy_so_far=%.3f cumulative_reward=%.3f",
        task_name.upper(),
        env.correct_predictions / max(1, env.step_count),
        env.cumulative_reward,
    )
    return predictions, list(env.ground_truth_labels), confidences


def main() -> Dict[str, object]:
    """Run the baseline across all tasks and persist the report."""

    logger.info("%s", "=" * 72)
    logger.info("FraudShield baseline inference")
    logger.info("%s", "=" * 72)

    env = FraudShieldEnvironment(data_path="data", seed=42)
    if not env.load_data():
        logger.error("FraudShield data could not be loaded from ./data")
        sys.exit(1)

    agent = build_default_agent()
    logger.info(
        "Agent mode: %s | API_BASE_URL=%s | MODEL_NAME=%s",
        getattr(agent, "name", agent.__class__.__name__),
        get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1"),
        get_env("MODEL_NAME", "MODELNAME", default="<offline-heuristic>"),
    )

    easy_predictions, easy_ground_truth, easy_confidences = run_task(env, agent, "easy")
    medium_predictions, medium_ground_truth, medium_confidences = run_task(env, agent, "medium")
    hard_predictions, hard_ground_truth, hard_confidences = run_task(env, agent, "hard")

    grading_result = FraudShieldGrader.grade_all_tasks(
        easy_predictions,
        easy_ground_truth,
        easy_confidences,
        medium_predictions,
        medium_ground_truth,
        medium_confidences,
        hard_predictions,
        hard_ground_truth,
        hard_confidences,
    )
    grading_result["metadata"] = {
        "agent_name": getattr(agent, "name", agent.__class__.__name__),
        "api_base_url": get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1"),
        "model_name": get_env("MODEL_NAME", "MODELNAME"),
        "seed": 42,
        "data_snapshot": env.data_loader.get_bundle_summary(),
        "tasks": {
            "easy": len(easy_ground_truth),
            "medium": len(medium_ground_truth),
            "hard": len(hard_ground_truth),
        },
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
