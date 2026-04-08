#!/usr/bin/env python3
"""Competition baseline inference for FraudShield.

This module provides the main entry point for evaluation:
1. Initialize environment with frozen data snapshot
2. Load agent (heuristic or LLM-powered)
3. Run all 3 task difficulties
4. Grade predictions against ground truth
5. Save results to fraudshield_baseline_results.json

Execution Modes:
  - Heuristic (offline): No external API, deterministic fraud rules
    Command: python inference.py
    Result: Baseline score (easy=1.0, medium=0.877, hard=0.721, final=0.866)
  
  - LLM (online): Calls OpenAI-compatible API with reasoning prompt
    Command: API_BASE_URL=... MODEL_NAME=... python inference.py
    Result: LLM reasoning + baseline grading

Output:
  - fraudshield_baseline_results.json: Complete grading report with:
    - Per-task scores (easy, medium, hard)
    - Final weighted score
    - Metadata (agent, model, seed, data snapshot)
    - Prediction traces (for replay/audit)

Logging:
  - INFO: Task progress, scores, file paths
  - ERROR: Data load failures, agent exceptions
  - EXCEPTION: Full traceback if inference fails

Usage Examples:
  # Heuristic baseline (no API needed)
  python inference.py

  # With LLM (requires API credentials)
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
  python inference.py

  # In Docker (PATH already set)
  docker run -e API_BASE_URL=... -e MODEL_NAME=... fraudshield:v0.2.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Tuple

from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import HeuristicFraudDetectionAgent, build_default_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "fraudshield_baseline_results.json"


def get_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from a list of aliases.
    
    Tries multiple variable names in order (useful for supporting different naming conventions).
    
    Args:
        *names: Environment variable names to check (in order of preference).
        default: Fallback value if none of the names are set.
    
    Returns:
        The first non-empty value found, or default if none matched.
    
    Example:
        api_url = get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1")
        model = get_env("MODEL_NAME", "MODELNAME", default="meta-llama/Llama-2-7b")
    """

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped_value = value.strip()
            if stripped_value:
                return stripped_value
    return default


def emit_event(event_name: str, **fields: object) -> None:
    """Print validator-friendly structured progress blocks to stdout."""

    parts = [f"[{event_name}]"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def build_resilient_agent() -> object:
    """Prefer the configured agent but never fail the baseline on init issues."""

    try:
        return build_default_agent()
    except Exception as exc:
        logger.warning(
            "Agent initialization failed: %s. Falling back to the deterministic heuristic agent.",
            exc,
        )
        return HeuristicFraudDetectionAgent()


def run_task(
    env: FraudShieldEnvironment,
    agent: object,
    task_name: str,
) -> Tuple[List[str], List[str], List[float], object, float]:
    """Run one task episode and capture the full prediction trace.
    
    This function executes a complete episode for a single task difficulty,
    collecting all predictions, confidences, and ground truth labels.
    
    Args:
        env: FraudShieldEnvironment instance (with data already loaded).
        agent: Agent object with decide(observation) method.
        task_name: Task difficulty ("easy", "medium", or "hard").
    
    Returns:
        Tuple containing:
        - predictions: List[str] of decisions ("fraud" or "legitimate")
        - ground_truth: List[str] of true labels
        - confidences: List[float] of confidence values [0.0, 1.0]
        - agent: Possibly updated agent if a fallback was needed
        - cumulative_reward: Total episode reward for the task
    
    Workflow:
        1. Call env.reset(task_name) to initialize episode
        2. Loop: agent.decide(obs) → env.step(action) → next obs
        3. Log progress each step
        4. Collect all decisions and ground truth
        5. Return predictions for grading
    
    Logging:
        - Task header with agent name
        - Progress every 10 steps (or at first/last)
        - Final accuracy and cumulative reward
    
    Example:
        preds, labels, confs = run_task(env, agent, "easy")
        print(f"Accuracy: {sum(p == l for p, l in zip(preds, labels)) / len(preds)}")
    """

    agent_name = getattr(agent, "name", agent.__class__.__name__)
    logger.info("START %s %s", task_name, agent_name)
    emit_event("START", task=task_name, agent=agent_name)

    reset_result = env.reset(task_name)

    observation = reset_result.observation
    predictions: List[str] = []
    confidences: List[float] = []
    fallback_agent: object | None = None

    while not env.is_done:
        try:
            action = agent.decide(observation)
        except Exception as exc:
            if fallback_agent is None:
                fallback_agent = HeuristicFraudDetectionAgent()
                logger.warning(
                    "Agent decision failed on task %s at step %s: %s. Switching to heuristic fallback.",
                    task_name,
                    env.step_count + 1,
                    exc,
                )
            agent = fallback_agent
            action = agent.decide(observation)

        predictions.append(action.decision.value)
        confidences.append(action.confidence)
        step_result = env.step(action)

        logger.info(
            "STEP %02d %s %.2f %+.2f",
            env.step_count,
            action.decision.value,
            action.confidence,
            step_result.reward.value,
        )
        emit_event(
            "STEP",
            task=task_name,
            step=env.step_count,
            decision=action.decision.value,
            confidence=f"{action.confidence:.2f}",
            reward=f"{step_result.reward.value:+.2f}",
        )

        observation = step_result.observation

    accuracy = env.correct_predictions / max(1, env.step_count)
    logger.info(
        "END %s %.3f %.3f",
        task_name.upper(),
        accuracy,
        env.cumulative_reward,
    )
    return predictions, list(env.ground_truth_labels), confidences, agent, env.cumulative_reward


def main() -> Dict[str, object]:
    """Run the baseline across all tasks and persist the report.
    
    This is the main entry point. It orchestrates the complete evaluation:
    1. Create environment and load frozen data snapshot
    2. Build agent (heuristic or LLM-powered)
    3. Run easy/medium/hard tasks sequentially
    4. Grade all predictions
    5. Save results to fraudshield_baseline_results.json
    
    Returns:
        Grading report dict with keys:
        - easy: {score, predictions, ground_truth, confidences}
        - medium: {...}
        - hard: {...}
        - final_score: Weighted average across all tasks
        - metadata: {agent_name, model_name, seed, data_snapshot, tasks}
    
    Error Handling:
        - Exits with code 1 if data fails to load
        - Exits with code 1 if inference crashes
        - Logs full exception traceback
    
    Side Effects:
        - Writes fraudshield_baseline_results.json to cwd
        - Logs task progress and scores
    
    Environment Variables:
        - API_BASE_URL: OpenAI-compatible API endpoint (for LLM mode)
        - MODEL_NAME: Model to use (for LLM mode)
        - (Both optional; heuristic mode runs offline if not set)
    
    Example:
        result = main()
        print(f"Final score: {result['final_score']:.4f}")
        print(f"Easy: {result['easy']['score']:.4f}")
    """

    logger.info("START FraudShield baseline inference")

    env = FraudShieldEnvironment(data_path="data", seed=42)
    if not env.load_data():
        logger.error("FraudShield data could not be loaded from ./data")
        sys.exit(1)

    agent = build_resilient_agent()
    logger.info(
        "Agent mode: %s | API_BASE_URL=%s | MODEL_NAME=%s",
        getattr(agent, "name", agent.__class__.__name__),
        getattr(agent, "api_base_url", get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1")),
        getattr(agent, "model_name", get_env("MODEL_NAME", "MODELNAME", default="<offline-heuristic>")),
    )

    easy_predictions, easy_ground_truth, easy_confidences, agent, easy_reward = run_task(env, agent, "easy")
    easy_result = FraudShieldGrader.grade_easy_task(easy_predictions, easy_ground_truth, easy_confidences)
    emit_event(
        "END",
        task="easy",
        score=f"{easy_result['score']:.4f}",
        reward=f"{easy_reward:.4f}",
        steps=len(easy_ground_truth),
    )

    medium_predictions, medium_ground_truth, medium_confidences, agent, medium_reward = run_task(env, agent, "medium")
    medium_result = FraudShieldGrader.grade_medium_task(
        medium_predictions,
        medium_ground_truth,
        medium_confidences,
    )
    emit_event(
        "END",
        task="medium",
        score=f"{medium_result['score']:.4f}",
        reward=f"{medium_reward:.4f}",
        steps=len(medium_ground_truth),
    )

    hard_predictions, hard_ground_truth, hard_confidences, agent, hard_reward = run_task(env, agent, "hard")
    hard_result = FraudShieldGrader.grade_hard_task(
        hard_predictions,
        hard_ground_truth,
        hard_confidences,
    )
    emit_event(
        "END",
        task="hard",
        score=f"{hard_result['score']:.4f}",
        reward=f"{hard_reward:.4f}",
        steps=len(hard_ground_truth),
    )

    final_score = (easy_result["score"] + medium_result["score"] + hard_result["score"]) / 3.0
    grading_result = {
        "final_score": float(final_score),
        "easy": easy_result,
        "medium": medium_result,
        "hard": hard_result,
        "breakdown": {
            "easy_weight": 1 / 3,
            "medium_weight": 1 / 3,
            "hard_weight": 1 / 3,
        },
    }
    grading_result["metadata"] = {
        "agent_name": getattr(agent, "name", agent.__class__.__name__),
        "api_base_url": getattr(agent, "api_base_url", get_env("API_BASE_URL", "APIBASEURL", default="https://router.huggingface.co/v1")),
        "model_name": getattr(agent, "model_name", get_env("MODEL_NAME", "MODELNAME")),
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
    logger.info("END FraudShield %.4f", grading_result["final_score"])
    emit_event(
        "END",
        task="overall",
        score=f"{grading_result['final_score']:.4f}",
        steps=len(easy_ground_truth) + len(medium_ground_truth) + len(hard_ground_truth),
    )

    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(grading_result, handle, indent=2)
    logger.info("Saved baseline report to %s", RESULTS_FILE)
    return grading_result


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
        logger.info("Baseline inference completed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("Baseline inference interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Baseline inference failed with exception: %s", exc)
        sys.exit(1)
