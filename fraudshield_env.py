"""FraudShield environment implementation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List

from data_loader import FraudDataLoader
from models import (
    DecisionEnum,
    EpisodeState,
    FraudCheckAction,
    FraudCheckObservation,
    ResetResult,
    Reward,
    StepResult,
    TaskDifficulty,
    TransactionData,
)


class FraudShieldEnvironment:
    """OpenEnv-compatible environment for e-commerce fraud review."""

    def __init__(self, data_path: str = "data", seed: int = 42):
        self.seed = seed
        self.data_loader = FraudDataLoader(data_path=data_path, seed=seed)
        self.data_loaded = False

        self.episode_id = ""
        self.current_task = TaskDifficulty.EASY
        self.step_count = 0
        self.current_transaction_idx = 0
        self.cumulative_reward = 0.0
        self.correct_predictions = 0
        self.is_done = False

        self.current_cases: List[Dict[str, Any]] = []
        self.ground_truth_labels: List[str] = []
        self.predictions: List[str] = []
        self.confidences: List[float] = []

        self.max_steps = {
            TaskDifficulty.EASY: 24,
            TaskDifficulty.MEDIUM: 36,
            TaskDifficulty.HARD: 48,
        }

    def load_data(self) -> bool:
        """Load the committed snapshot or rebuild it from the local public source CSV."""

        self.data_loaded = self.data_loader.load_data()
        return self.data_loaded

    def load_kaggle_data(self) -> bool:
        """Backward-compatible wrapper for the previous method name."""

        return self.load_data()

    def ensure_data_loaded(self) -> None:
        """Load data on demand so server startup can stay simple."""

        if not self.data_loaded and not self.load_data():
            raise RuntimeError("FraudShield data bundle could not be loaded.")

    def reset(self, task: str = "easy") -> ResetResult:
        """Start a fresh episode for a given task difficulty."""

        self.ensure_data_loaded()

        self.episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self.current_task = TaskDifficulty(task)
        self.step_count = 0
        self.current_transaction_idx = 0
        self.cumulative_reward = 0.0
        self.correct_predictions = 0
        self.is_done = False
        self.predictions = []
        self.confidences = []

        self.current_cases = self.data_loader.get_task_cases(task)
        self.ground_truth_labels = [case["label"] for case in self.current_cases]
        self.max_steps[self.current_task] = len(self.current_cases)

        observation = self._get_observation()
        info = {
            "episode_id": self.episode_id,
            "task": task,
            "task_focus": observation.historical_context.get("task_focus") if observation.historical_context else None,
            "data_snapshot": self.data_loader.get_bundle_summary(),
            "max_steps": self.max_steps[self.current_task],
            "num_transactions": len(self.current_cases),
            "fraud_count": sum(1 for label in self.ground_truth_labels if label == "fraud"),
            "legitimate_count": sum(1 for label in self.ground_truth_labels if label == "legitimate"),
        }
        return ResetResult(observation=observation, info=info)

    def step(self, action: FraudCheckAction) -> StepResult:
        """Evaluate one agent action and return the next observation."""

        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_case = self.current_cases[self.current_transaction_idx]
        expected_transaction_id = current_case["transaction_id"]
        wrong_transaction_id = action.transaction_id != expected_transaction_id
        ground_truth = current_case["label"]
        risk_score = float(current_case["risk_score"])
        business_cost = float(current_case["business_cost"])

        predicted_label = action.decision.value
        is_correct = predicted_label == ground_truth and not wrong_transaction_id

        reward_value, confidence_penalty, reward_reason = self._calculate_reward(
            predicted_label=predicted_label,
            ground_truth=ground_truth,
            confidence=action.confidence,
            risk_score=risk_score,
            business_cost=business_cost,
            wrong_transaction_id=wrong_transaction_id,
        )

        if is_correct:
            self.correct_predictions += 1

        self.predictions.append(predicted_label)
        self.confidences.append(action.confidence)
        self.cumulative_reward += reward_value
        self.step_count += 1
        self.current_transaction_idx += 1
        self.is_done = self.current_transaction_idx >= len(self.current_cases)

        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            is_correct=is_correct,
            ground_truth=DecisionEnum(ground_truth),
            confidence_penalty=confidence_penalty,
            business_impact=business_cost,
        )

        observation = self._get_terminal_observation() if self.is_done else self._get_observation()
        info = {
            "step": self.step_count,
            "accuracy_so_far": round(self.correct_predictions / self.step_count, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "expected_transaction_id": expected_transaction_id,
            "wrong_transaction_id": wrong_transaction_id,
            "risk_score": risk_score,
            "business_cost": business_cost,
        }
        return StepResult(observation=observation, reward=reward, done=self.is_done, info=info)

    def state(self) -> EpisodeState:
        """Return the current episode state."""

        return EpisodeState(
            episode_id=self.episode_id,
            task_name=self.current_task,
            step_count=self.step_count,
            transactions_evaluated=self.current_transaction_idx,
            cumulative_reward=self.cumulative_reward,
            correct_predictions=self.correct_predictions,
            is_done=self.is_done,
            max_steps=self.max_steps[self.current_task],
        )

    def _calculate_reward(
        self,
        predicted_label: str,
        ground_truth: str,
        confidence: float,
        risk_score: float,
        business_cost: float,
        wrong_transaction_id: bool,
    ) -> tuple[float, float, str]:
        """Apply dense reward shaping with business-cost sensitivity."""

        is_fraud_case = ground_truth == "fraud"
        predicted_fraud = predicted_label == "fraud"

        if is_fraud_case and predicted_fraud:
            base_reward = 0.68 + (0.16 * business_cost)
        elif not is_fraud_case and not predicted_fraud:
            base_reward = 0.54 + (0.06 * (1.2 - min(business_cost, 1.2)))
        elif is_fraud_case and not predicted_fraud:
            base_reward = -0.72 - (0.14 * business_cost)
        else:
            base_reward = -0.48 - (0.08 * business_cost)

        target_confidence = risk_score if is_fraud_case else (1.0 - risk_score)
        confidence_penalty = 0.12 - abs(confidence - target_confidence) * 0.24
        if predicted_label != ground_truth:
            confidence_penalty -= 0.04 + (confidence * 0.06)
        if wrong_transaction_id:
            confidence_penalty -= 0.10

        reward_value = max(-1.0, min(1.0, base_reward + confidence_penalty))
        reason_bits = [
            f"predicted={predicted_label}",
            f"actual={ground_truth}",
            f"target_confidence={target_confidence:.2f}",
        ]
        if wrong_transaction_id:
            reason_bits.append("action referenced the wrong transaction_id")
        reward_reason = ", ".join(reason_bits)
        return reward_value, confidence_penalty, reward_reason

    def _get_observation(self) -> FraudCheckObservation:
        """Return the current task observation."""

        current_case = self.current_cases[self.current_transaction_idx]
        return FraudCheckObservation(
            transaction_id=current_case["transaction_id"],
            transaction_data=TransactionData(**current_case["transaction_data"]),
            task_name=self.current_task,
            episode_step=self.step_count + 1,
            historical_context=current_case["historical_context"],
        )

    def _get_terminal_observation(self) -> FraudCheckObservation:
        """Return a terminal observation once the episode completes."""

        terminal_transaction = TransactionData(
            amount=0.0,
            seller_id="TERMINAL",
            buyer_id="TERMINAL",
            item_category="none",
            item_price=0.0,
            shipping_address="XX",
            seller_account_age_days=0,
            buyer_account_age_days=0,
            payment_method="none",
            device_country="XX",
            timestamp=datetime.utcnow().isoformat(),
            is_repeat_buyer=False,
            seller_avg_rating=0.0,
            num_seller_reviews=0,
            previous_fraud_flags=0,
            shipping_speed="none",
            amount_percentile=0.0,
            seller_chargeback_rate_30d=0.0,
            buyer_disputes_90d=0,
            shared_device_accounts_24h=0,
            same_address_orders_24h=0,
        )
        return FraudCheckObservation(
            transaction_id="TERMINAL",
            transaction_data=terminal_transaction,
            task_name=self.current_task,
            episode_step=max(1, self.step_count),
            historical_context={"episode_done": True},
        )
