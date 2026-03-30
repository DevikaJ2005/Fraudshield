"""Typed models for the FraudShield OpenEnv environment."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DecisionEnum(str, Enum):
    """Fraud review decision emitted by the agent."""

    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class TaskDifficulty(str, Enum):
    """Supported task difficulties."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FraudCheckAction(BaseModel):
    """Action taken by the reviewing agent for a single transaction."""

    model_config = ConfigDict(use_enum_values=False)

    transaction_id: str = Field(..., description="Unique transaction identifier.")
    decision: DecisionEnum = Field(..., description="Predicted fraud label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence assigned to the prediction.",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Short explanation supporting the decision.",
    )


class TransactionData(BaseModel):
    """Observed transaction details exposed to the agent."""

    amount: float = Field(..., ge=0.0, description="Checkout amount in USD.")
    seller_id: str = Field(..., description="Seller account identifier.")
    buyer_id: str = Field(..., description="Buyer account identifier.")
    item_category: str = Field(..., description="Primary product category.")
    item_price: float = Field(..., ge=0.0, description="Listed item price in USD.")
    shipping_address: str = Field(..., description="Shipping destination country code.")
    seller_account_age_days: int = Field(..., ge=0, description="Seller age in days.")
    buyer_account_age_days: int = Field(..., ge=0, description="Buyer age in days.")
    payment_method: str = Field(..., description="Normalized payment method label.")
    device_country: str = Field(..., description="Country inferred from the device/IP.")
    timestamp: str = Field(..., description="ISO-8601 transaction timestamp.")
    is_repeat_buyer: bool = Field(..., description="Whether buyer purchased from seller before.")
    seller_avg_rating: float = Field(..., ge=0.0, le=5.0, description="Seller rating from 0 to 5.")
    num_seller_reviews: int = Field(..., ge=0, description="Published seller review count.")
    previous_fraud_flags: int = Field(..., ge=0, description="Historical fraud flags on related accounts.")
    shipping_speed: str = Field(..., description="Requested shipping speed.")
    amount_percentile: float = Field(..., ge=0.0, le=100.0, description="Spend percentile versus the marketplace.")
    seller_chargeback_rate_30d: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Seller chargeback ratio over the last 30 days.",
    )
    buyer_disputes_90d: int = Field(..., ge=0, description="Buyer disputes filed in the last 90 days.")
    shared_device_accounts_24h: int = Field(
        ...,
        ge=0,
        description="Accounts seen on the same device in the last 24 hours.",
    )
    same_address_orders_24h: int = Field(
        ...,
        ge=0,
        description="Orders sent to the same address in the last 24 hours.",
    )


class FraudCheckObservation(BaseModel):
    """Observation returned to the agent at each environment step."""

    model_config = ConfigDict(use_enum_values=False)

    transaction_id: str = Field(..., description="Transaction identifier for the current case.")
    transaction_data: TransactionData = Field(..., description="Structured transaction facts.")
    task_name: TaskDifficulty = Field(..., description="Active task difficulty.")
    episode_step: int = Field(..., ge=1, description="One-based position in the episode.")
    historical_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Rolling marketplace context relevant to this transaction.",
    )


class Reward(BaseModel):
    """Reward signal returned after each agent action."""

    model_config = ConfigDict(use_enum_values=False)

    value: float = Field(..., ge=-1.0, le=1.0, description="Dense reward for the action.")
    reason: str = Field(..., description="Human-readable summary of the reward calculation.")
    is_correct: bool = Field(..., description="Whether the prediction matched the hidden label.")
    ground_truth: DecisionEnum = Field(..., description="Hidden ground-truth label revealed after acting.")
    confidence_penalty: float = Field(
        ...,
        ge=-0.3,
        le=0.3,
        description="Calibration adjustment based on confidence quality.",
    )
    business_impact: float = Field(
        ...,
        ge=0.5,
        le=2.0,
        description="Relative business cost multiplier for the current case.",
    )


class EpisodeState(BaseModel):
    """Serializable snapshot of the current episode."""

    model_config = ConfigDict(use_enum_values=False)

    episode_id: str = Field(..., description="Unique identifier for the episode.")
    task_name: TaskDifficulty = Field(..., description="Current task difficulty.")
    step_count: int = Field(..., ge=0, description="Number of actions executed so far.")
    transactions_evaluated: int = Field(..., ge=0, description="Transactions completed so far.")
    cumulative_reward: float = Field(..., description="Total reward accumulated this episode.")
    correct_predictions: int = Field(..., ge=0, description="Correct predictions made so far.")
    is_done: bool = Field(..., description="Whether the episode has reached a terminal state.")
    max_steps: int = Field(..., ge=1, description="Maximum number of allowed steps in the task.")


class StepResult(BaseModel):
    """Result returned by ``step()``."""

    observation: FraudCheckObservation = Field(..., description="Next observation.")
    reward: Reward = Field(..., description="Reward assigned to the submitted action.")
    done: bool = Field(..., description="Whether the episode is complete.")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Supplementary metadata.")


class ResetResult(BaseModel):
    """Result returned by ``reset()``."""

    observation: FraudCheckObservation = Field(..., description="Initial observation.")
    info: Dict[str, Any] = Field(..., description="Episode metadata.")
