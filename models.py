"""
FraudShield Models
Type-safe Pydantic models for OpenEnv environment
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class DecisionEnum(str, Enum):
    """Agent fraud decision"""
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class TaskDifficulty(str, Enum):
    """Task difficulty level"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ============================================================================
# ACTION MODELS
# ============================================================================

class FraudCheckAction(BaseModel):
    """
    Agent action: Examine a transaction and decide if it's fraudulent
    
    Fields:
    - transaction_id: Unique identifier for the transaction
    - decision: "fraud" or "legitimate"
    - confidence: 0.0 (not sure) to 1.0 (very sure)
    - reasoning: Brief explanation of the decision
    """
    transaction_id: str = Field(..., description="Unique transaction ID")
    decision: DecisionEnum = Field(..., description="Fraud or legitimate")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ..., 
        min_length=10, 
        max_length=500,
        description="Explanation of the decision"
    )

    class Config:
        use_enum_values = False


# ============================================================================
# OBSERVATION MODELS
# ============================================================================

class TransactionData(BaseModel):
    """Details of a transaction to evaluate"""
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD")
    seller_id: str = Field(..., description="Seller account ID")
    buyer_id: str = Field(..., description="Buyer account ID")
    item_category: str = Field(..., description="Product category")
    item_price: float = Field(..., ge=0.0, description="Item listed price")
    shipping_address: str = Field(..., description="Country/region code")
    seller_account_age_days: int = Field(..., ge=0, description="Days since seller registered")
    buyer_account_age_days: int = Field(..., ge=0, description="Days since buyer registered")
    payment_method: str = Field(..., description="visa, mastercard, paypal, etc")
    device_country: str = Field(..., description="Country of device making purchase")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    is_repeat_buyer: bool = Field(..., description="Has buyer purchased from seller before")
    seller_avg_rating: float = Field(..., ge=0.0, le=5.0, description="Seller rating 0-5")
    num_seller_reviews: int = Field(..., ge=0, description="Number of seller reviews")
    previous_fraud_flags: int = Field(..., ge=0, description="Times this account was flagged")


class FraudCheckObservation(BaseModel):
    """
    Current state observation: A transaction for the agent to evaluate
    """
    transaction_id: str = Field(..., description="Transaction identifier")
    transaction_data: TransactionData = Field(..., description="Full transaction details")
    task_name: TaskDifficulty = Field(..., description="Difficulty level")
    episode_step: int = Field(..., ge=1, description="Current step number")
    historical_context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context from previous transactions"
    )

    class Config:
        use_enum_values = False


# ============================================================================
# REWARD MODELS
# ============================================================================

class Reward(BaseModel):
    """
    Reward signal for agent's action
    """
    value: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0,
        description="Reward value from -1.0 to +1.0"
    )
    reason: str = Field(..., description="Why this reward was given")
    is_correct: bool = Field(..., description="Whether decision was correct")
    ground_truth: DecisionEnum = Field(..., description="Actual label (fraud/legitimate)")
    confidence_penalty: float = Field(
        ..., 
        ge=-0.2, 
        le=0.2,
        description="Adjustment based on confidence alignment"
    )


# ============================================================================
# STATE MODELS
# ============================================================================

class EpisodeState(BaseModel):
    """
    Current episode state: Metadata about the ongoing episode
    """
    episode_id: str = Field(..., description="Unique episode ID")
    task_name: TaskDifficulty = Field(..., description="Current task difficulty")
    step_count: int = Field(..., ge=0, description="Total steps in episode")
    transactions_evaluated: int = Field(..., ge=0, description="Transactions reviewed so far")
    cumulative_reward: float = Field(..., description="Sum of all rewards in episode")
    is_done: bool = Field(..., description="Is episode complete?")
    max_steps: int = Field(..., ge=1, description="Max steps allowed in this task")

    class Config:
        use_enum_values = False


# ============================================================================
# STEP RESULT MODELS
# ============================================================================

class StepResult(BaseModel):
    """
    Result of calling step() on the environment
    """
    observation: FraudCheckObservation = Field(..., description="New observation")
    reward: Reward = Field(..., description="Reward for action")
    done: bool = Field(..., description="Is episode done?")
    info: Optional[Dict[str, Any]] = Field(None, description="Debugging info")


# ============================================================================
# RESET RESULT MODELS
# ============================================================================

class ResetResult(BaseModel):
    """
    Result of calling reset() on the environment
    """
    observation: FraudCheckObservation = Field(..., description="Initial observation")
    info: Dict[str, Any] = Field(..., description="Episode metadata")
