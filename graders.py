"""Deterministic graders for FraudShield tasks."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


class FraudShieldGrader:
    """Task graders returning scores in the strict range (0.0, 1.0)."""

    STRICT_SCORE_EPSILON = 1e-4

    @staticmethod
    def _validate(predictions: List[str], ground_truth: List[str], confidences: List[float]) -> bool:
        return bool(predictions) and len(predictions) == len(ground_truth) == len(confidences)

    @staticmethod
    def _to_labels(predictions: List[str], ground_truth: List[str]) -> tuple[List[int], List[int]]:
        y_true = [1 if label == "fraud" else 0 for label in ground_truth]
        y_pred = [1 if label == "fraud" else 0 for label in predictions]
        return y_true, y_pred

    @staticmethod
    def _score_confidence(predictions: List[str], confidences: List[float]) -> List[float]:
        return [confidence if pred == "fraud" else 1.0 - confidence for pred, confidence in zip(predictions, confidences)]

    @staticmethod
    def _strict_score(score: float) -> float:
        """Clamp task scores to the open interval required by the submission validator."""

        return float(
            max(
                FraudShieldGrader.STRICT_SCORE_EPSILON,
                min(1.0 - FraudShieldGrader.STRICT_SCORE_EPSILON, score),
            )
        )

    @staticmethod
    def _classification_metrics(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float],
    ) -> Dict[str, float]:
        y_true, y_pred = FraudShieldGrader._to_labels(predictions, ground_truth)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        calibration_targets = np.asarray(y_true, dtype=float)
        confidence_scores = np.asarray(FraudShieldGrader._score_confidence(predictions, confidences), dtype=float)
        brier = float(np.mean((confidence_scores - calibration_targets) ** 2))
        calibration_score = max(0.0, 1.0 - brier)
        try:
            roc_auc = float(roc_auc_score(y_true, confidence_scores))
        except ValueError:
            roc_auc = 0.5
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "roc_auc": roc_auc,
            "calibration_score": float(calibration_score),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    @staticmethod
    def grade_easy_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """Easy task emphasizes obvious-case accuracy and false-positive control."""

        if not FraudShieldGrader._validate(predictions, ground_truth, confidences):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid predictions"}

        metrics = FraudShieldGrader._classification_metrics(predictions, ground_truth, confidences)
        score = (
            metrics["accuracy"] * 0.45
            + metrics["f1_score"] * 0.25
            + metrics["recall"] * 0.15
            + metrics["specificity"] * 0.15
        )
        return FraudShieldGrader._build_response("easy", score, metrics, ground_truth)

    @staticmethod
    def grade_medium_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """Medium task rewards balanced classification and calibrated confidence."""

        if not FraudShieldGrader._validate(predictions, ground_truth, confidences):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid predictions"}

        metrics = FraudShieldGrader._classification_metrics(predictions, ground_truth, confidences)
        score = (
            metrics["f1_score"] * 0.40
            + metrics["roc_auc"] * 0.30
            + metrics["precision"] * 0.15
            + metrics["calibration_score"] * 0.15
        )
        return FraudShieldGrader._build_response("medium", score, metrics, ground_truth)

    @staticmethod
    def grade_hard_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """Hard task weights fraud capture, precision, and ranking quality."""

        if not FraudShieldGrader._validate(predictions, ground_truth, confidences):
            return {"score": FraudShieldGrader._strict_score(0.0), "reason": "Invalid predictions"}

        metrics = FraudShieldGrader._classification_metrics(predictions, ground_truth, confidences)
        score = (
            metrics["recall"] * 0.35
            + metrics["precision"] * 0.20
            + metrics["f1_score"] * 0.20
            + metrics["roc_auc"] * 0.15
            + metrics["calibration_score"] * 0.10
        )
        return FraudShieldGrader._build_response("hard", score, metrics, ground_truth)

    @staticmethod
    def grade_task(
        task_name: str,
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """Dispatch to the correct task grader."""

        graders = {
            "easy": FraudShieldGrader.grade_easy_task,
            "medium": FraudShieldGrader.grade_medium_task,
            "hard": FraudShieldGrader.grade_hard_task,
        }
        if task_name not in graders:
            raise ValueError(f"Unknown task: {task_name}")
        return graders[task_name](predictions, ground_truth, confidences)

    @staticmethod
    def grade_all_tasks(
        easy_predictions: List[str],
        easy_ground_truth: List[str],
        easy_confidences: List[float],
        medium_predictions: List[str],
        medium_ground_truth: List[str],
        medium_confidences: List[float],
        hard_predictions: List[str],
        hard_ground_truth: List[str],
        hard_confidences: List[float],
    ) -> Dict[str, Any]:
        """Grade all three tasks and aggregate them into one final score."""

        easy_result = FraudShieldGrader.grade_easy_task(easy_predictions, easy_ground_truth, easy_confidences)
        medium_result = FraudShieldGrader.grade_medium_task(
            medium_predictions,
            medium_ground_truth,
            medium_confidences,
        )
        hard_result = FraudShieldGrader.grade_hard_task(hard_predictions, hard_ground_truth, hard_confidences)

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
    def _build_response(
        task_name: str,
        score: float,
        metrics: Dict[str, float],
        ground_truth: List[str],
    ) -> Dict[str, Any]:
        return {
            "score": FraudShieldGrader._strict_score(score),
            "task": task_name,
            "metrics": metrics,
            "num_transactions": len(ground_truth),
            "fraud_count": sum(1 for label in ground_truth if label == "fraud"),
            "legitimate_count": sum(1 for label in ground_truth if label == "legitimate"),
        }
