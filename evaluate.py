"""Evaluation entrypoint for FraudShield trainable agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from config import ExperimentConfig
from environment import FraudShieldTextEnvironment
from llm_agent import build_default_agent
from utils import ensure_dir, moving_average, save_json, seed_everything


def evaluate_agent(config: ExperimentConfig) -> dict[str, Any]:
    """Run fixed-task evaluations and collect comparison metrics."""

    seed_everything(config.seed)
    text_env = FraudShieldTextEnvironment(config.environment, config.reward_weights)
    agent = build_default_agent()
    task_rows = []
    reward_traces: dict[str, list[float]] = {}
    for task in config.evaluation.tasks:
        prompt = text_env.reset(task=task)
        done = False
        rewards: list[float] = []
        final_info: dict[str, Any] | None = None
        while not done:
            action = agent.decide(text_env.current_observation)
            response_text = json.dumps(
                {
                    "action_type": "decide" if action.action_type.value == "resolve_case" else "investigate",
                    "investigation_target": action.action_type.value,
                    "decision": "fraud" if getattr(action, "resolution", None) and action.resolution.value in {"block", "hold", "escalate"} else "legitimate",
                    "confidence": 0.8,
                    "reasoning": action.reasoning or "Evaluation rollout step.",
                }
            )
            step = text_env.step(response_text)
            prompt = step.next_prompt
            done = step.done
            rewards.append(step.reward)
            final_info = step.info
        reward_traces[task] = rewards
        state = final_info["state"] if final_info else {}
        env_reward = final_info["env_reward"] if final_info else {}
        task_rows.append(
            {
                "task": task,
                "total_reward": round(sum(rewards), 4),
                "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
                "success_rate": 1.0 if env_reward.get("is_correct") else 0.0,
                "resolved_cases": len(state.get("resolved_case_ids", [])),
                "token_usage_estimate": sum(len(str(value)) for value in rewards),
            }
        )
    return {"tasks": task_rows, "reward_traces": reward_traces}


def save_evaluation_artifacts(report: dict[str, Any], config: ExperimentConfig) -> None:
    """Persist evaluation metrics and plots."""

    plots_dir = ensure_dir(config.evaluation.plots_dir)
    rewards = [row["total_reward"] for row in report["tasks"]]
    moving = moving_average(rewards, window=2)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o", label="reward")
    plt.plot(range(1, len(moving) + 1), moving, marker="x", label="moving_avg_reward")
    plt.xticks(range(1, len(rewards) + 1), [row["task"] for row in report["tasks"]])
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "evaluation_rewards.png")
    plt.close()
    save_json(report, Path(config.training.output_dir) / "evaluation_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FraudShield trainable agents.")
    parser.add_argument("--config", default="configs/colab_qlora_grpo.json", help="Path to experiment config JSON.")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    report = evaluate_agent(config)
    save_evaluation_artifacts(report, config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
