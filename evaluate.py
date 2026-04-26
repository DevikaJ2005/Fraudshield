"""Evaluation entrypoint for FraudShield trainable agents."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from config import ExperimentConfig
from utils import ensure_dir, moving_average, save_json, seed_everything


def _run_inference(extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run ``inference.py`` with a controlled environment and return its report."""

    env_vars = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "API_KEY", "OPENAI_API_KEY", "API_BASE_URL", "MODEL_NAME", "LOCAL_MODEL_PATH"):
        env_vars.pop(key, None)
    if extra_env:
        env_vars.update(extra_env)
    subprocess.run(
        ["python", "inference.py"],
        check=True,
        capture_output=True,
        text=True,
        env=env_vars,
    )
    with open("fraudshield_baseline_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_agent(config: ExperimentConfig) -> dict[str, Any]:
    """Compare heuristic baseline against the trained local checkpoint."""

    seed_everything(config.seed)
    trained_model_path = str(Path(config.training.output_dir) / "trained_policy")

    baseline_results = _run_inference()
    trained_results = _run_inference({"LOCAL_MODEL_PATH": trained_model_path})

    comparison_rows = []
    win_count = 0
    for task in config.evaluation.tasks:
        baseline_score = float(baseline_results[task]["score"])
        trained_score = float(trained_results[task]["score"])
        if trained_score > baseline_score:
            win_count += 1
        comparison_rows.append(
            {
                "task": task,
                "baseline_score": baseline_score,
                "trained_score": trained_score,
                "delta": trained_score - baseline_score,
            }
        )

    return {
        "baseline": {
            "easy": baseline_results["easy"]["score"],
            "medium": baseline_results["medium"]["score"],
            "hard": baseline_results["hard"]["score"],
            "final_score": baseline_results["final_score"],
            "agent_metadata": baseline_results.get("metadata", {}),
        },
        "trained": {
            "easy": trained_results["easy"]["score"],
            "medium": trained_results["medium"]["score"],
            "hard": trained_results["hard"]["score"],
            "final_score": trained_results["final_score"],
            "agent_metadata": trained_results.get("metadata", {}),
            "local_model_path": trained_model_path,
        },
        "comparison": comparison_rows,
        "success_rate": win_count / max(1, len(config.evaluation.tasks)),
        "preference_score": trained_results["final_score"] - baseline_results["final_score"],
        "before_after": {
            "base_model_final": baseline_results["final_score"],
            "trained_model_final": trained_results["final_score"],
        },
    }


def save_evaluation_artifacts(report: dict[str, Any], config: ExperimentConfig) -> None:
    """Persist evaluation metrics and plots."""

    plots_dir = ensure_dir(config.evaluation.plots_dir)

    baseline_scores = [row["baseline_score"] for row in report["comparison"]]
    trained_scores = [row["trained_score"] for row in report["comparison"]]
    deltas = [row["delta"] for row in report["comparison"]]
    labels = [row["task"] for row in report["comparison"]]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(baseline_scores) + 1), baseline_scores, marker="o", label="baseline")
    plt.plot(range(1, len(trained_scores) + 1), trained_scores, marker="o", label="trained")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("task score")
    plt.title("FraudShield before vs after")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "before_after_scores.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(deltas) + 1), deltas, marker="o", label="score_delta")
    plt.plot(range(1, len(deltas) + 1), moving_average(deltas, window=2), marker="x", label="moving_avg_delta")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("score delta")
    plt.title("FraudShield score improvement by task")
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
