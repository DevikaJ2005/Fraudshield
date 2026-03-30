"""
FraudShield Graders
Evaluation logic for 3 tasks (Easy, Medium, Hard)
"""

from typing import List, Dict, Any
from models import DecisionEnum, TaskDifficulty
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


class FraudShieldGrader:
    """
    Evaluates agent performance on fraud detection tasks
    3 Tasks: EASY, MEDIUM, HARD with increasing difficulty
    """

    @staticmethod
    def grade_easy_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Grade EASY task - Clear fraud signals"""
        if not predictions or len(predictions) != len(ground_truth):
            return {"score": 0.0, "reason": "Invalid predictions"}
        
        y_true = [1 if label == "fraud" else 0 for label in ground_truth]
        y_pred = [1 if pred == "fraud" else 0 for pred in predictions]
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        score = (accuracy * 0.4) + (f1 * 0.3) + (recall * 0.2) + (max(0, 1 - false_positive_rate * 2) * 0.1)
        score = min(1.0, max(0.0, score))
        
        return {
            "score": float(score),
            "task": "easy",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "false_positive_rate": float(false_positive_rate),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
            "num_transactions": len(predictions),
            "fraud_count": sum(1 for x in ground_truth if x == "fraud"),
            "legitimate_count": sum(1 for x in ground_truth if x == "legitimate"),
        }

    @staticmethod
    def grade_medium_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Grade MEDIUM task - Mixed signals, ROC-AUC focus"""
        if not predictions or len(predictions) != len(ground_truth):
            return {"score": 0.0, "reason": "Invalid predictions"}
        
        y_true = [1 if label == "fraud" else 0 for label in ground_truth]
        y_pred = [1 if pred == "fraud" else 0 for pred in predictions]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            adjusted_confidences = []
            for i, pred in enumerate(predictions):
                conf = confidences[i]
                if pred == "fraud":
                    adjusted_confidences.append(conf)
                else:
                    adjusted_confidences.append(1.0 - conf)
            roc_auc = roc_auc_score(y_true, adjusted_confidences)
        except:
            roc_auc = 0.5
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        score = (f1 * 0.6) + (roc_auc * 0.4)
        score = min(1.0, max(0.0, score))
        
        return {
            "score": float(score),
            "task": "medium",
            "metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
            "num_transactions": len(predictions),
            "fraud_count": sum(1 for x in ground_truth if x == "fraud"),
            "legitimate_count": sum(1 for x in ground_truth if x == "legitimate"),
        }

    @staticmethod
    def grade_hard_task(
        predictions: List[str],
        ground_truth: List[str],
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Grade HARD task - Complex patterns, ring fraud detection"""
        if not predictions or len(predictions) != len(ground_truth):
            return {"score": 0.0, "reason": "Invalid predictions"}
        
        y_true = [1 if label == "fraud" else 0 for label in ground_truth]
        y_pred = [1 if pred == "fraud" else 0 for pred in predictions]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        score = (recall * 0.4) + (precision * 0.3) + (f1 * 0.2) + (max(0, 1 - false_negative_rate * 3) * 0.1)
        score = min(1.0, max(0.0, score))
        
        return {
            "score": float(score),
            "task": "hard",
            "metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "false_negative_rate": float(false_negative_rate),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
            "num_transactions": len(predictions),
            "fraud_count": sum(1 for x in ground_truth if x == "fraud"),
            "legitimate_count": sum(1 for x in ground_truth if x == "legitimate"),
        }

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
        """Grade all 3 tasks and compute final score"""
        easy_result = FraudShieldGrader.grade_easy_task(
            easy_predictions, easy_ground_truth, easy_confidences
        )
        medium_result = FraudShieldGrader.grade_medium_task(
            medium_predictions, medium_ground_truth, medium_confidences
        )
        hard_result = FraudShieldGrader.grade_hard_task(
            hard_predictions, hard_ground_truth, hard_confidences
        )
        
        final_score = (easy_result["score"] + medium_result["score"] + hard_result["score"]) / 3.0
        
        return {
            "final_score": float(final_score),
            "easy": easy_result,
            "medium": medium_result,
            "hard": hard_result,
            "breakdown": {
                "easy_weight": 0.33,
                "medium_weight": 0.33,
                "hard_weight": 0.34,
            },
        }
