"""Typed models for the FraudShield OpenEnv environment.

This module defines all request/response models using Pydantic v2 for:
- Type validation (enforced at API and environment boundaries)
- JSON serialization (FastAPI/HTTP compatibility)
- Schema generation (OpenAPI docs, IDE type hints)
- IDE autocompletion (full typing information)

Model Hierarchy:
  Input Models (Agent → Environment):
    - FraudCheckAction: Fraud decision submitted by agent

  Output Models (Environment → Agent):
    - FraudCheckObservation: Transaction facts + history
    - Reward: Dense reward + metadata
    - EpisodeState: Full episode snapshot
    - StepResult: Complete step output (observation + reward + done)
    - ResetResult: Episode initialization output

  Enums (Controlled vocabularies):
    - DecisionEnum: "fraud" or "legitimate"
    - TaskDifficulty: "easy", "medium", or "hard"

  Data Structures:
    - TransactionData: 20 fields describing a single transaction
    - Historical context: Prior observations, rolling statistics, etc.

Validation Rules:
  - Amount/Confidence/Ratings: Bounded ranges (ge/le constraints)
  - Text fields: Length constraints (min_length/max_length)
  - Enums: Limited to valid values
  - Timestamps: ISO-8601 format (enforced by environment)

JSON Serialization:
  All models use Pydantic's model_dump(mode='json') for HTTP responses.
  Enums serialized as strings (e.g., {"decision": "fraud"}).

Usage:
    from models import FraudCheckAction, FraudCheckObservation
    
    # Parse incoming action from API request body
    action = FraudCheckAction.model_validate_json(request_body)
    
    # Create response observation
    obs = FraudCheckObservation(
        transaction_id="txn_001",
        transaction_data=TransactionData(...),
        task_name=TaskDifficulty.EASY,
        episode_step=1,
        historical_context={}
    )
    response = obs.model_dump(mode='json')  # Serialize to JSON dict
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DecisionEnum(str, Enum):
    """Fraud review decision emitted by the agent.
    
    Valid values:
        - "fraud": Transaction is fraudulent (should be rejected)
        - "legitimate": Transaction is legitimate (should be approved)
    """

    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class TaskDifficulty(str, Enum):
    """Supported task difficulties.
    
    Tasks differ in transaction count, fraud/legitimate overlap, and signal clarity:
    
        - "easy" (45 transactions): Clear separability, obvious fraud markers
        - "medium" (50 transactions): Mixed signals, calibration matters
        - "hard" (65 transactions): High overlap, coordinated abuse, edge cases
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FraudCheckAction(BaseModel):
    """Action taken by the reviewing agent for a single transaction.
    
    The agent observes a transaction and submits a decision with confidence.
    The environment returns a reward and the next observation.
    
    Attributes:
        transaction_id: Unique transaction identifier (matches obs.transaction_id).
        decision: Fraud label ("fraud" or "legitimate").
        confidence: Confidence in the decision as a probability [0.0, 1.0].
            - 1.0 = completely confident
            - 0.5 = maximal uncertainty
            - 0.0 = completely confident in the opposite class
            Reward includes calibration penalty: |confidence - is_correct| matters.
        reasoning: Brief explanation supporting the decision (10-500 chars).
            Used for ablation studies, not by environment reward function.
    
    Validation:
        - decision: Must be valid DecisionEnum value
        - confidence: Must be in [0.0, 1.0] (float)
        - reasoning: Must be 10-500 character string
    
    Example:
        action = FraudCheckAction(
            transaction_id="txn_001",
            decision=DecisionEnum.FRAUD,
            confidence=0.92,
            reasoning="Seller account created 2 days ago, requested overnight shipping for electronics."
        )
    """

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
    """Observed transaction details exposed to the agent.
    
    This model represents a single e-commerce transaction with 20 fields covering:
    - Transaction basics: amount, item, pricing
    - Seller context: age, rating, reputation, chargeback rate
    - Buyer context: age, history, disputes, account sharing
    - Fraud signals: geographical mismatches, velocity, device analysis
    
    All fields are derived from the frozen Kaggle snapshot and enriched with
    synthetic marketplace context (seller age, disputes, etc.) for realism.
    
    Attributes:
        amount: Checkout total in USD (float ≥ 0.0).
        seller_id: Unique seller account identifier (string, hashable).
        buyer_id: Unique buyer account identifier (string, hashable).
        item_category: Primary product category (e.g., "Electronics", "Apparel").
        item_price: Listed item price in USD before markup (float ≥ 0.0).
        shipping_address: 2-letter country code (e.g., "US", "GB", "FR").
        seller_account_age_days: Days seller account has existed (int ≥ 0).
            - 0-7: Very new seller (high fraud risk)
            - 7-90: New seller (moderate risk)
            - 90+: Established seller (lower risk)
        buyer_account_age_days: Days buyer account has existed (int ≥ 0).
        payment_method: Normalized label ("card", "paypal", "bank_transfer", etc.).
        device_country: Country inferred from device/IP geolocation (2-letter code).
        timestamp: ISO-8601 transaction timestamp (string).
        is_repeat_buyer: Whether buyer has purchased from this seller before (bool).
        seller_avg_rating: Seller average rating from 0.0 to 5.0 (float 0-5).
        num_seller_reviews: Number of published seller reviews (int ≥ 0).
        previous_fraud_flags: Historical fraud flags on related accounts (int ≥ 0).
            - Includes seller account, buyer account, and shared devices
        shipping_speed: Requested shipping strategy ("standard", "expedited", "overnight").
        amount_percentile: Transaction amount percentile vs marketplace (float 0-100).
            - 100 = highest value transaction (high fraud risk if unusual)
            - 50 = median transaction (baseline risk)
            - 1 = lowest value transaction
        seller_chargeback_rate_30d: Seller chargeback ratio in last 30 days (float 0-1).
            - 0.0 = no chargebacks (very safe)
            - 0.1+ = concerning chargeback rate (elevated risk)
        buyer_disputes_90d: Disputes filed by this buyer in last 90 days (int ≥ 0).
            - 0 = no disputes (trustworthy)
            - 3+ = dispute pattern (potential malicious buyer)
        shared_device_accounts_24h: Accounts seen on same device in last 24h (int ≥ 0).
            - 1 = only this account (normal)
            - 3+ = multiple accounts (potential fraud ring)
        same_address_orders_24h: Orders shipped to same address in last 24h (int ≥ 0).
            - 1 = only this order (normal)
            - 5+ = velocity attack pattern
    
    Validation:
        - amount, item_price: Must be ≥ 0.0
        - seller_avg_rating: Must be 0.0-5.0
        - seller_chargeback_rate_30d: Must be 0.0-1.0
        - amount_percentile: Must be 0.0-100.0
        - All `*_days` fields: Must be ≥ 0
        - All `*_count` fields: Must be ≥ 0
    
    Example:
        txn = TransactionData(
            amount=150.00,
            seller_id="seller_123",
            buyer_id="buyer_456",
            item_category="Electronics",
            item_price=140.00,
            shipping_address="US",
            seller_account_age_days=2,  # Very new!
            buyer_account_age_days=15,
            payment_method="card",
            device_country="NG",  # Mismatch with shipping_address!
            timestamp="2023-10-15T14:30:00Z",
            is_repeat_buyer=False,
            seller_avg_rating=0.0,  # No history
            num_seller_reviews=0,
            previous_fraud_flags=3,
            shipping_speed="overnight",
            amount_percentile=99.5,  # Very high value
            seller_chargeback_rate_30d=0.15,
            buyer_disputes_90d=2,
            shared_device_accounts_24h=4,  # Ring pattern
            same_address_orders_24h=6  # Velocity attack
        )
    
    Note:
        All numeric values are rounded/discretized for interpretability.
        Geographic codes follow ISO 3166-1 alpha-2 standard.
    """

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
    """Observation returned to the agent at each environment step.
    
    This is the primary input to the agent's policy. It contains the current
    transaction details and contextual information needed to make a fraud decision.
    
    Attributes:
        transaction_id: Unique identifier for the current transaction (matches action.transaction_id).
        transaction_data: Complete transaction details (20 fields).
        task_name: Current task difficulty ("easy", "medium", "hard").
        episode_step: One-based step number in the episode (1, 2, 3, ...).
        historical_context: Optional dict with rolling marketplace statistics
            (e.g., fraud rate in last hour, merchant category patterns).
            May be None in early implementations.
    
    Example:
        obs = FraudCheckObservation(
            transaction_id="txn_001",
            transaction_data=TransactionData(...),
            task_name=TaskDifficulty.EASY,
            episode_step=1,
            historical_context={"fraud_rate_1h": 0.02}
        )
    """

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
    """Reward signal returned after each agent action.
    
    The reward is dense (every step signals quality), business-cost-sensitive,
    and includes calibration feedback (penalizing overconfidence).
    
    Attributes:
        value: Dense reward [-1.0, 1.0] indicating action quality.
            - +1.0: Correct detection of fraud with perfect confidence
            - +0.8: Correct approval of legitimate with good confidence
            - -0.5: False positive (rejected legitimate transaction)
            - -1.0: False negative (approved fraudulent transaction)
            Calibration penalty applied: rewards decrease if confidence mismatches accuracy.
        reason: Human-readable summary explaining the reward calculation.
        is_correct: Whether the prediction matched the ground truth label.
        ground_truth: The hidden ground truth label (fraud or legitimate).
            Revealed only after the agent acts (learning signal).
        confidence_penalty: Calibration adjustment [-0.3, 0.3] based on confidence quality.
            - Positive: Agent was overconfident in correct decision (small penalty)
            - Negative: Agent was underconfident (penalty for not committing)
            - 0.0: Confidence matched accuracy perfectly
        business_impact: Relative business cost multiplier for this case [0.5, 2.0].
            - Cases with high customer value: business_impact ~ 2.0 (error costs more)
            - Cases with low risk: business_impact ~ 0.5 (error matters less)
    
    Example:
        reward = Reward(
            value=0.95,
            reason="Correct fraud detection with high confidence (0.92) - excellent action.",
            is_correct=True,
            ground_truth=DecisionEnum.FRAUD,
            confidence_penalty=-0.05,
            business_impact=1.8
        )
    """

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
    """Serializable snapshot of the current episode.
    
    This model captures the complete episode state at any point in time.
    Useful for debugging, replay, and monitoring agent progress.
    
    Attributes:
        episode_id: Unique identifier for this episode (string).
        task_name: Current task difficulty.
        step_count: Number of actions submitted so far (0-based, incremented after each step).
        transactions_evaluated: Number of transactions completed (same as step_count).
        cumulative_reward: Sum of all reward.value fields from step 1 to now.
        correct_predictions: Number of steps where is_correct=True.
        is_done: Whether episode has reached terminal state (all transactions reviewed).
        max_steps: Maximum allowed steps for this task (45, 50, or 65).
    
    Derived Metrics:
        - accuracy: correct_predictions / step_count (if step_count > 0)
        - avg_reward: cumulative_reward / step_count (if step_count > 0)
        - progress: step_count / max_steps
    
    Example:
        state = EpisodeState(
            episode_id="ep_abc123",
            task_name=TaskDifficulty.EASY,
            step_count=5,
            transactions_evaluated=5,
            cumulative_reward=2.45,
            correct_predictions=4,
            is_done=False,
            max_steps=45
        )
        accuracy = state.correct_predictions / state.step_count  # 0.8 (80%)
    """

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
    """Result returned by ``step()``.
    
    This model wraps all outputs from a single environment step, including
    the next observation, reward signal, termination flag, and metadata.
    
    Attributes:
        observation: Next observation (or final state if done=True).
        reward: Reward for the action just submitted.
        done: Whether episode is complete (all transactions reviewed or error).
        info: Dict with optional supplementary data (debugging, logging).
    
    Example:
        result = env.step(action)
        obs = result.observation
        reward = result.reward
        if result.done:
            print(f"Episode complete! Final score: {reward.value}")
        else:
            print(f"Step {obs.episode_step} / {max_steps}")
    """

    observation: FraudCheckObservation = Field(..., description="Next observation.")
    reward: Reward = Field(..., description="Reward assigned to the submitted action.")
    done: bool = Field(..., description="Whether the episode is complete.")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Supplementary metadata.")


class ResetResult(BaseModel):
    """Result returned by ``reset()``.
    
    This model initializes a fresh episode with the requested task difficulty.
    
    Attributes:
        observation: Initial observation (first transaction).
        info: Episode metadata (task, episode_id, max_steps, etc.).
    
    Example:
        result = env.reset(TaskDifficulty.EASY)
        obs = result.observation
        print(f"Episode {result.info['episode_id']} started. Task: {result.info['task']}")
    """

    observation: FraudCheckObservation = Field(..., description="Initial observation.")
    info: Dict[str, Any] = Field(..., description="Episode metadata.")
