"""Deterministic graders for the FraudShield FraudOps workflow."""

from __future__ import annotations

from typing import Any, Dict


class FraudShieldGrader:
    """Task graders returning strict 0-1 scores from workflow summaries."""

    STRICT_SCORE_EPSILON = 1e-4

    @staticmethod
    def _strict_score(score: float) -> float:
        return float(
            max(
                FraudShieldGrader.STRICT_SCORE_EPSILON,
                min(1.0 - FraudShieldGrader.STRICT_SCORE_EPSILON, score),
            )
        )

    @staticmethod
    def _validate(summary: Dict[str, Any]) -> bool:
        metrics = summary.get("metrics", {})
        return bool(summary.get("case_summaries")) and isinstance(metrics, dict)

    @staticmethod
    def grade_easy_task(summary: Dict[str, Any]) -> Dict[str, Any]:
        if not FraudShieldGrader._validate(summary):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid episode summary"}

        metrics = summary["metrics"]
        score = (
            metrics["resolution_accuracy"] * 0.55
            + metrics["workflow_completion"] * 0.25
            + metrics["efficiency"] * 0.20
        )
        return FraudShieldGrader._build_response("easy", score, summary)

    @staticmethod
    def grade_medium_task(summary: Dict[str, Any]) -> Dict[str, Any]:
        if not FraudShieldGrader._validate(summary):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid episode summary"}

        metrics = summary["metrics"]
        score = (
            metrics["resolution_accuracy"] * 0.40
            + metrics["policy_compliance"] * 0.25
            + metrics["evidence_coverage"] * 0.20
            + metrics["workflow_completion"] * 0.10
            + metrics["efficiency"] * 0.05
        )
        return FraudShieldGrader._build_response("medium", score, summary)

    @staticmethod
    def grade_hard_task(summary: Dict[str, Any]) -> Dict[str, Any]:
        if not FraudShieldGrader._validate(summary):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid episode summary"}

        metrics = summary["metrics"]
        score = (
            metrics["resolution_accuracy"] * 0.30
            + metrics["policy_compliance"] * 0.20
            + metrics["link_consistency"] * 0.20
            + metrics["evidence_coverage"] * 0.15
            + metrics["workflow_completion"] * 0.10
            + metrics["efficiency"] * 0.05
        )
        return FraudShieldGrader._build_response("hard", score, summary)

    @staticmethod
    def grade_task(task_name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        graders = {
            "easy": FraudShieldGrader.grade_easy_task,
            "medium": FraudShieldGrader.grade_medium_task,
            "hard": FraudShieldGrader.grade_hard_task,
        }
        if task_name not in graders:
            raise ValueError(f"Unknown task: {task_name}")
        return graders[task_name](summary)

    @staticmethod
    def grade_all_tasks(
        easy_summary: Dict[str, Any],
        medium_summary: Dict[str, Any],
        hard_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        easy_result = FraudShieldGrader.grade_easy_task(easy_summary)
        medium_result = FraudShieldGrader.grade_medium_task(medium_summary)
        hard_result = FraudShieldGrader.grade_hard_task(hard_summary)

        final_score = (easy_result["score"] + medium_result["score"] + hard_result["score"]) / 3.0
        return {
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

    @staticmethod
    def _build_response(task_name: str, score: float, summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": FraudShieldGrader._strict_score(score),
            "task": task_name,
            "metrics": summary.get("metrics", {}),
            "step_count": summary.get("step_count"),
            "max_steps": summary.get("max_steps"),
            "invalid_action_count": summary.get("invalid_action_count"),
            "redundant_action_count": summary.get("redundant_action_count"),
            "note_spam_count": summary.get("note_spam_count"),
            "case_summaries": summary.get("case_summaries", []),
        }
