#!/usr/bin/env python3
"""Run a simple CLI walkthrough of the FraudShield environment."""

from __future__ import annotations

import argparse
import json

from fraudshield_env import FraudShieldEnvironment
from llm_agent import SnapshotCalibratedFraudDetectionAgent


def run_demo(task: str, max_actions: int) -> None:
    env = FraudShieldEnvironment(data_path="data", seed=42)
    env.load_data()
    agent = SnapshotCalibratedFraudDetectionAgent()

    observation = env.reset(task).observation
    print("=" * 72)
    print(f"FraudShield Demo | task={task}")
    print("=" * 72)

    while not env.is_done and env.step_count < max_actions:
        print(
            f"\nStep {env.step_count + 1} | case={observation.case_id} | screen={observation.current_screen.value}"
        )
        print(f"Allowed actions: {[action.value for action in observation.allowed_actions]}")
        print(f"Visible panels: {observation.visible_panels}")
        print(f"Case summary: {json.dumps(observation.case_summary.model_dump(mode='json'), indent=2)}")
        if observation.revealed_evidence:
            print("Revealed evidence keys:", sorted(observation.revealed_evidence.keys()))

        action = agent.decide(observation)
        print("Action:", action.model_dump(mode="json"))
        step_result = env.step(action)
        print(
            "Reward:",
            {
                "value": step_result.reward.value,
                "reason": step_result.reward.reason,
                "correct": step_result.reward.is_correct,
                "policy_compliant": step_result.reward.policy_compliant,
            },
        )
        observation = step_result.observation

    print("\nEpisode summary:")
    print(json.dumps(env.get_episode_report(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short FraudShield walkthrough.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()
    run_demo(task=args.task, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
