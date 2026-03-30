"""
FraudShield Environment - Kaggle Edition
Fraud detection environment using real Credit Card dataset
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import random
from models import (
    FraudCheckAction, FraudCheckObservation, Reward, EpisodeState,
    StepResult, ResetResult, TransactionData, DecisionEnum, TaskDifficulty
)
from data_loader import KaggleDataLoader


class FraudShieldEnvironment:
    """
    Fraud Detection OpenEnv using real Kaggle Credit Card data
    Agent evaluates real transactions and predicts if fraudulent.
    Three difficulty levels: easy, medium, hard.
    """

    def __init__(self, data_path: str = "data", seed: int = 42):
        """Initialize environment with Kaggle data"""
        self.seed = seed
        random.seed(seed)
        
        # Load data
        self.data_loader = KaggleDataLoader(data_path)
        self.data_loaded = False
        
        # Episode state
        self.episode_id: str = ""
        self.current_task: TaskDifficulty = TaskDifficulty.EASY
        self.step_count: int = 0
        self.current_transaction_idx: int = 0
        self.cumulative_reward: float = 0.0
        self.is_done: bool = False
        
        # Task data
        self.current_transactions: List[Dict[str, Any]] = []
        self.ground_truth_labels: List[str] = []
        
        # Agent predictions
        self.predictions: List[str] = []
        self.confidences: List[float] = []
        
        # Max steps per task
        self.max_steps = {
            TaskDifficulty.EASY: 100,
            TaskDifficulty.MEDIUM: 150,
            TaskDifficulty.HARD: 250,
        }

    def load_kaggle_data(self) -> bool:
        """Load Kaggle dataset"""
        if self.data_loader.load_data():
            self.data_loaded = True
            return True
        return False

    def reset(self, task: str = "easy") -> ResetResult:
        """Reset environment for a new episode"""
        if not self.data_loaded:
            raise RuntimeError("Kaggle data not loaded. Call load_kaggle_data() first")
        
        self.episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self.current_task = TaskDifficulty(task)
        self.step_count = 0
        self.current_transaction_idx = 0
        self.cumulative_reward = 0.0
        self.is_done = False
        self.predictions = []
        self.confidences = []
        
        # Load transactions for this task
        self._load_task_data(task)
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "episode_id": self.episode_id,
            "task": task,
            "max_steps": self.max_steps[self.current_task],
            "num_transactions": len(self.current_transactions),
            "fraud_count": sum(1 for x in self.ground_truth_labels if x == "fraud"),
        }
        
        return ResetResult(observation=observation, info=info)

    def step(self, action: FraudCheckAction) -> StepResult:
        """Agent takes action and gets reward"""
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        # Record prediction
        self.predictions.append(action.decision.value)
        self.confidences.append(action.confidence)
        
        # Calculate reward
        ground_truth = self.ground_truth_labels[self.current_transaction_idx]
        is_correct = action.decision.value == ground_truth
        
        # Reward logic
        base_reward = 1.0 if is_correct else -0.5
        
        # Confidence penalty
        if is_correct:
            confidence_penalty = (action.confidence - 0.5) * 0.2
        else:
            confidence_penalty = (1.0 - action.confidence) * 0.2
        
        reward_value = base_reward + confidence_penalty
        reward_value = max(-1.0, min(1.0, reward_value))
        
        reward = Reward(
            value=reward_value,
            reason=f"Predicted {action.decision.value}, actual was {ground_truth}",
            is_correct=is_correct,
            ground_truth=DecisionEnum(ground_truth),
            confidence_penalty=confidence_penalty
        )
        
        self.cumulative_reward += reward_value
        self.step_count += 1
        self.current_transaction_idx += 1
        
        # Check if done
        max_steps = self.max_steps[self.current_task]
        if self.current_transaction_idx >= len(self.current_transactions) or self.step_count >= max_steps:
            self.is_done = True
        
        # Get next observation
        if self.is_done:
            observation = self._get_terminal_observation()
        else:
            observation = self._get_observation()
        
        info = {
            "step": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "is_correct": is_correct,
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=self.is_done,
            info=info
        )

    def state(self) -> EpisodeState:
        """Get current episode state"""
        return EpisodeState(
            episode_id=self.episode_id,
            task_name=self.current_task,
            step_count=self.step_count,
            transactions_evaluated=self.current_transaction_idx,
            cumulative_reward=self.cumulative_reward,
            is_done=self.is_done,
            max_steps=self.max_steps[self.current_task],
        )

    def _load_task_data(self, task: str):
        """Load transaction data for task difficulty from Kaggle"""
        easy, medium, hard, easy_labels, medium_labels, hard_labels = \
            self.data_loader.get_split_by_difficulty()
        
        if task == "easy":
            self.current_transactions = easy
            self.ground_truth_labels = easy_labels
        elif task == "medium":
            self.current_transactions = medium
            self.ground_truth_labels = medium_labels
        elif task == "hard":
            self.current_transactions = hard
            self.ground_truth_labels = hard_labels
        else:
            raise ValueError(f"Unknown task: {task}")

    def _get_observation(self) -> FraudCheckObservation:
        """Get observation for current transaction"""
        if self.current_transaction_idx >= len(self.current_transactions):
            return self._get_terminal_observation()
        
        txn_dict = self.current_transactions[self.current_transaction_idx]
        
        # Remove PCA features (they're for reference, not for observation)
        txn_dict_clean = {k: v for k, v in txn_dict.items() if k != "pca_features"}
        
        transaction_data = TransactionData(**txn_dict_clean)
        
        observation = FraudCheckObservation(
            transaction_id=txn_dict["transaction_id"],
            transaction_data=transaction_data,
            task_name=self.current_task,
            episode_step=self.step_count + 1,
            historical_context=None,
        )
        
        return observation

    def _get_terminal_observation(self) -> FraudCheckObservation:
        """Return terminal observation when episode ends"""
        terminal_txn = TransactionData(
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
            timestamp=datetime.now().isoformat(),
            is_repeat_buyer=False,
            seller_avg_rating=0.0,
            num_seller_reviews=0,
            previous_fraud_flags=0,
        )
        
        return FraudCheckObservation(
            transaction_id="TERMINAL",
            transaction_data=terminal_txn,
            task_name=self.current_task,
            episode_step=self.step_count,
            historical_context={"episode_done": True},
        )
