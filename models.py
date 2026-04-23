"""Typed models for the FraudShield FraudOps environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskDifficulty(str, Enum):
    """Supported graded tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionTypeEnum(str, Enum):
    """Explicit enterprise tool actions available to the agent."""

    REVIEW_TRANSACTION = "review_transaction"
    FETCH_CUSTOMER_PROFILE = "fetch_customer_profile"
    FETCH_MERCHANT_PROFILE = "fetch_merchant_profile"
    FETCH_NETWORK_GRAPH = "fetch_network_graph"
    CHECK_POLICY = "check_policy"
    ADD_CASE_NOTE = "add_case_note"
    RESOLVE_CASE = "resolve_case"


class ResolutionEnum(str, Enum):
    """Final case routing actions."""

    APPROVE = "approve"
    BLOCK = "block"
    HOLD = "hold"
    REQUEST_DOCS = "request_docs"
    ESCALATE = "escalate"


class CaseScreenEnum(str, Enum):
    """Simulated enterprise apps in the FraudOps workflow."""

    QUEUE = "Queue"
    CASE_CONSOLE = "Case Console"
    CUSTOMER_PROFILE = "Customer Profile"
    MERCHANT_PROFILE = "Merchant Profile"
    POLICY_ESCALATION = "Policy & Escalation"


class QueueCaseCard(BaseModel):
    """Visible queue item shown before deeper investigation."""

    case_id: str = Field(..., description="Unique review case identifier.")
    priority: str = Field(..., description="Queue priority label.")
    queue_reason: str = Field(..., description="Short visible reason the case entered the queue.")
    visible_risk_band: str = Field(..., description="Queue-only coarse risk label.")
    status: str = Field(..., description="Case status shown in the queue.")
    linked_case_ids: List[str] = Field(default_factory=list, description="Related cases if visible.")


class CaseSummary(BaseModel):
    """Current high-level summary for the active case."""

    case_id: str = Field(..., description="Active case identifier.")
    status: str = Field(..., description="Current workflow status.")
    queue_reason: str = Field(..., description="Short queue reason shown to the agent.")
    visible_risk_band: str = Field(..., description="Coarse risk band visible without hidden labels.")
    amount_usd: float = Field(..., ge=0.0, description="Transaction amount visible from the queue or console.")
    merchant_region: str = Field(..., description="Shipping region or country code.")
    evidence_collected: List[str] = Field(default_factory=list, description="Evidence bundle keys already revealed.")
    note_added: bool = Field(..., description="Whether a case note has already been written.")


class FraudCheckAction(BaseModel):
    """Action submitted by an agent to the FraudOps environment."""

    model_config = ConfigDict(use_enum_values=False)

    case_id: str = Field(..., description="Target case identifier for the action.")
    action_type: ActionTypeEnum = Field(..., description="Enterprise tool action to execute.")
    note_text: Optional[str] = Field(
        default=None,
        max_length=600,
        description="Case note text when action_type='add_case_note'.",
    )
    resolution: Optional[ResolutionEnum] = Field(
        default=None,
        description="Final routing outcome when action_type='resolve_case'.",
    )
    reasoning: str = Field(
        default="",
        max_length=500,
        description="Short rationale for the selected action.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "FraudCheckAction":
        reasoning = self.reasoning.strip()
        note_text = self.note_text.strip() if self.note_text else None

        if self.action_type == ActionTypeEnum.ADD_CASE_NOTE:
            if not note_text or len(note_text) < 12:
                raise ValueError("note_text must be at least 12 characters when action_type='add_case_note'")
            if self.resolution is not None:
                raise ValueError("add_case_note actions must not include resolution")
        elif self.action_type == ActionTypeEnum.RESOLVE_CASE:
            if self.resolution is None:
                raise ValueError("resolution is required when action_type='resolve_case'")
            if len(reasoning) < 12:
                raise ValueError("reasoning must be at least 12 characters when action_type='resolve_case'")
            if note_text is not None:
                raise ValueError("resolve_case actions must not include note_text")
        else:
            if note_text is not None or self.resolution is not None:
                raise ValueError("tool actions must not include note_text or resolution")

        self.reasoning = reasoning
        self.note_text = note_text
        return self


class FraudCheckObservation(BaseModel):
    """Observation returned at reset and after every step."""

    model_config = ConfigDict(use_enum_values=False)

    case_id: str = Field(..., description="Currently active case identifier.")
    task_name: TaskDifficulty = Field(..., description="Current task difficulty.")
    current_screen: CaseScreenEnum = Field(..., description="Current enterprise app screen.")
    visible_panels: List[str] = Field(..., description="Currently visible panels on the active screen.")
    revealed_evidence: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Evidence bundles revealed for the active case.",
    )
    linked_case_ids: List[str] = Field(default_factory=list, description="Related case identifiers.")
    remaining_steps: int = Field(..., ge=0, description="Remaining total step budget for the episode.")
    remaining_sla: int = Field(..., ge=0, description="Remaining SLA budget before penalties grow.")
    note_required: bool = Field(..., description="Whether the current case still requires a note before resolution.")
    allowed_actions: List[ActionTypeEnum] = Field(..., description="Actions currently considered valid.")
    queue_items: List[QueueCaseCard] = Field(default_factory=list, description="Queue view across all episode cases.")
    case_summary: CaseSummary = Field(..., description="Summary of the active case.")
    episode_step: int = Field(..., ge=0, description="Current 1-based step count within the episode.")
    app_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra app metadata such as workflow hints or policy flags already visible.",
    )


class Reward(BaseModel):
    """Dense reward returned for every step."""

    model_config = ConfigDict(use_enum_values=False)

    value: float = Field(..., ge=-1.0, le=1.0, description="Step reward in the closed interval [-1, 1].")
    reason: str = Field(..., description="Human-readable explanation for the reward assignment.")
    action_type: ActionTypeEnum = Field(..., description="Action family that produced the reward.")
    case_id: str = Field(..., description="Case affected by the action.")
    action_cost: float = Field(default=0.0, description="Explicit cost applied to the action.")
    sla_penalty: float = Field(default=0.0, description="Penalty applied for burning SLA budget.")
    evidence_key: Optional[str] = Field(default=None, description="Evidence key affected by the action, if any.")
    resolution: Optional[ResolutionEnum] = Field(default=None, description="Resolution submitted, if any.")
    ground_truth_resolution: Optional[ResolutionEnum] = Field(
        default=None,
        description="Hidden correct resolution once a final decision has been made.",
    )
    is_correct: Optional[bool] = Field(default=None, description="Whether the final case routing was correct.")
    policy_compliant: Optional[bool] = Field(
        default=None,
        description="Whether the final routing also complied with revealed policy constraints.",
    )
    anti_hacking_triggered: bool = Field(
        default=False,
        description="Whether the reward reflects anti-hacking or anti-spam penalties.",
    )


class EpisodeState(BaseModel):
    """Full state snapshot returned by ``state()``."""

    model_config = ConfigDict(use_enum_values=False)

    episode_id: str = Field(..., description="Current episode identifier.")
    task_name: TaskDifficulty = Field(..., description="Current task difficulty.")
    current_screen: CaseScreenEnum = Field(..., description="Current app screen.")
    active_case_id: str = Field(..., description="Currently focused case.")
    step_count: int = Field(..., ge=0, description="Number of actions taken so far.")
    remaining_steps: int = Field(..., ge=0, description="Remaining total step budget.")
    remaining_sla: int = Field(..., ge=0, description="Remaining SLA budget.")
    cumulative_reward: float = Field(..., description="Total reward accumulated this episode.")
    is_done: bool = Field(..., description="Whether the episode has terminated.")
    resolved_case_ids: List[str] = Field(default_factory=list, description="Case IDs already resolved.")
    unresolved_case_ids: List[str] = Field(default_factory=list, description="Case IDs still open.")
    notes_written_by_case: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of notes written for each case.",
    )
    evidence_keys_by_case: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Revealed evidence bundle keys per case.",
    )
    policy_checked_case_ids: List[str] = Field(
        default_factory=list,
        description="Case IDs where the policy tool has been consulted.",
    )
    resolution_by_case: Dict[str, ResolutionEnum] = Field(
        default_factory=dict,
        description="Submitted resolutions for already resolved cases.",
    )
    invalid_action_count: int = Field(default=0, ge=0, description="Number of invalid-order actions taken.")
    redundant_action_count: int = Field(default=0, ge=0, description="Number of redundant fetch/note actions taken.")


class StepResult(BaseModel):
    """Environment step result."""

    observation: FraudCheckObservation = Field(..., description="Next observation after the submitted action.")
    reward: Reward = Field(..., description="Reward assigned to the submitted action.")
    done: bool = Field(..., description="Whether the episode terminated after this step.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra runtime diagnostics.")


class ResetResult(BaseModel):
    """Environment reset result."""

    observation: FraudCheckObservation = Field(..., description="Initial observation for the new episode.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Episode metadata.")
