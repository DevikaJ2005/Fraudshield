#!/usr/bin/env python3
"""Showcase the multi-step investigation workflow in FraudShield."""

from __future__ import annotations

import argparse

from fraudshield_env import FraudShieldEnvironment
from llm_agent import AgenticHeuristicFraudDetectionAgent


def run_demo(task: str, max_decisions: int) -> None:
    env = FraudShieldEnvironment(data_path="data", seed=42)
    env.load_data()
    agent = AgenticHeuristicFraudDetectionAgent()

    reset_result = env.reset(task)
    observation = reset_result.observation
    decisions_seen = 0

    print(f"Episode: {reset_result.info['episode_id']}")
    print(f"Task: {task}")
    print(f"Max decisions to show: {max_decisions}")
    print()

    while not env.is_done and decisions_seen < max_decisions:
        print(f"Case: {observation.transaction_id}")
        print(f"Stage: {observation.case_stage}")
        print(f"Visible summary: {observation.visible_signal_summary}")
        print(f"Available investigations: {[target.value for target in observation.available_investigations]}")
        print(f"Revealed evidence: {sorted(observation.revealed_evidence.keys())}")

        action = agent.decide(observation)
        step_result = env.step(action)

        if action.action_type.value == "investigate":
            print(
                f"  Investigation -> {action.investigation_target.value} | "
                f"reward={step_result.reward.value:+.3f} | "
                f"stage={step_result.observation.case_stage}"
            )
        else:
            decisions_seen += 1
            print(
                f"  Decision -> {action.decision.value} @ {action.confidence:.2f} | "
                f"reward={step_result.reward.value:+.3f} | "
                f"correct={step_result.reward.is_correct}"
            )
            print(f"  Ground truth: {step_result.reward.ground_truth}")

        observation = step_result.observation
        print()

    print("Demo complete.")
    print(
        f"Actions taken={env.state().step_count}, "
        f"decisions={env.state().transactions_evaluated}, "
        f"investigations={env.state().investigations_used}, "
        f"cumulative_reward={env.state().cumulative_reward:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short FraudShield agentic demo.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--max-decisions", type=int, default=3)
    args = parser.parse_args()
    run_demo(task=args.task, max_decisions=args.max_decisions)


if __name__ == "__main__":  # pragma: no cover
    main()
