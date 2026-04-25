"""FraudShield partial-observability environment implementation."""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

from data_loader import FraudDataLoader
from models import (
    ActionTypeEnum,
    CaseScreenEnum,
    CaseSummary,
    EpisodeState,
    FraudCheckAction,
    FraudCheckObservation,
    QueueCaseCard,
    ResetResult,
    ResolutionEnum,
    Reward,
    StepResult,
    TaskDifficulty,
)

TASK_CONFIG: Dict[TaskDifficulty, Dict[str, Any]] = {
    TaskDifficulty.EASY: {
        "source_task": "easy",
        "num_cases": 1,
        "max_steps": 6,
        "sla_limit": 5,
        "ideal_steps": 3,
        "investigation_budget": 1,
        "minimum_fetches_for_bonus": 1,
        "description": "Single low-noise case with strong visible cues and one fetch budget.",
    },
    TaskDifficulty.MEDIUM: {
        "source_task": "medium",
        "num_cases": 1,
        "max_steps": 8,
        "sla_limit": 6,
        "ideal_steps": 5,
        "investigation_budget": 2,
        "minimum_fetches_for_bonus": 1,
        "description": "Single mixed-signal case that requires at least one investigation before routing.",
    },
    TaskDifficulty.HARD: {
        "source_task": "hard",
        "num_cases": 2,
        "max_steps": 14,
        "sla_limit": 11,
        "ideal_steps": 9,
        "investigation_budget": 3,
        "minimum_fetches_for_bonus": 1,
        "description": "Two misleading linked cases where graph evidence is usually required.",
    },
}

FETCH_ACTIONS = {
    ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
    ActionTypeEnum.FETCH_MERCHANT_PROFILE,
    ActionTypeEnum.FETCH_NETWORK_GRAPH,
    ActionTypeEnum.CHECK_POLICY,
}


class FraudShieldEnvironment:
    """OpenEnv-compatible fraud-investigation environment."""

    def __init__(self, data_path: str = "data", seed: int = 42):
        self.seed = seed
        self.data_loader = FraudDataLoader(data_path=data_path, seed=seed)
        self.data_loaded = False

        self.episode_id = ""
        self.current_task = TaskDifficulty.EASY
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.is_done = False
        self.current_screen = CaseScreenEnum.QUEUE
        self.active_case_id = ""

        self.workflow_cases: Dict[str, Dict[str, Any]] = {}
        self.case_state: Dict[str, Dict[str, Any]] = {}
        self.case_order: List[str] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.invalid_action_count = 0
        self.redundant_action_count = 0
        self.note_spam_count = 0
        self.case_counts = {task: TASK_CONFIG[task]["num_cases"] for task in TaskDifficulty}
        self.max_steps = {task: TASK_CONFIG[task]["max_steps"] for task in TaskDifficulty}
        self.sla_limit = {task: TASK_CONFIG[task]["sla_limit"] for task in TaskDifficulty}
        self.last_episode_summary: Dict[str, Any] = {}

    def load_data(self) -> bool:
        """Load the deterministic committed snapshot."""

        self.data_loaded = self.data_loader.load_data()
        return self.data_loaded

    def load_kaggle_data(self) -> bool:
        """Backward-compatible alias for older validation scripts."""

        return self.load_data()

    def ensure_data_loaded(self) -> None:
        """Load data lazily for local and remote execution."""

        if not self.data_loaded and not self.load_data():
            raise RuntimeError("FraudShield data bundle could not be loaded.")

    def reset(self, task: str = "easy") -> ResetResult:
        """Start a new fraud-investigation episode."""

        self.ensure_data_loaded()

        self.current_task = TaskDifficulty(task)
        config = TASK_CONFIG[self.current_task]

        self.episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.is_done = False
        self.current_screen = CaseScreenEnum.QUEUE
        self.invalid_action_count = 0
        self.redundant_action_count = 0
        self.note_spam_count = 0
        self.audit_log = []
        self.last_episode_summary = {}

        self.workflow_cases = self._build_workflow_cases(self.current_task)
        self.case_order = list(self.workflow_cases.keys())
        self.active_case_id = self.case_order[0]
        self.case_state = {
            case_id: {
                "status": "triage",
                "reviewed": False,
                "revealed_evidence": {},
                "note_count": 0,
                "notes": [],
                "policy_checked": False,
                "resolution": None,
                "resolved": False,
                "resolution_correct": False,
                "policy_compliant": False,
                "invalid_actions": 0,
                "redundant_actions": 0,
                "anti_hacking_hits": 0,
                "action_history": [],
                "fetches_used": 0,
                "fetch_budget_remaining": config["investigation_budget"],
            }
            for case_id in self.case_order
        }

        info = {
            "episode_id": self.episode_id,
            "task": self.current_task.value,
            "num_cases": config["num_cases"],
            "max_steps": config["max_steps"],
            "sla_limit": config["sla_limit"],
            "investigation_budget": config["investigation_budget"],
            "description": config["description"],
            "workflow_views": [screen.value for screen in CaseScreenEnum],
            "data_snapshot": self.data_loader.get_bundle_summary(),
        }
        return ResetResult(observation=self._build_observation(), info=info)

    def step(self, action: FraudCheckAction) -> StepResult:
        """Apply a single investigation or resolution action."""

        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if action.case_id not in self.workflow_cases:
            reward = self._apply_action_outcome(
                action=action,
                case_id=self.active_case_id or self.case_order[0],
                base_value=-0.35,
                reason=f"Unknown case_id '{action.case_id}'.",
                valid_action=False,
                anti_hacking=True,
            )
            return self._build_step_result(reward, {"valid_action": False, "error": "unknown_case"})

        if action.action_type != ActionTypeEnum.REVIEW_TRANSACTION and action.case_id != self.active_case_id:
            reward = self._apply_action_outcome(
                action=action,
                case_id=action.case_id,
                base_value=-0.18,
                reason="Only review_transaction may switch focus to another case.",
                valid_action=False,
            )
            return self._build_step_result(reward, {"valid_action": False, "error": "inactive_case"})

        case_id = action.case_id
        state = self.case_state[case_id]

        if state["resolved"] and action.action_type != ActionTypeEnum.REVIEW_TRANSACTION:
            reward = self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.22,
                reason="Resolved cases cannot be modified; move to an open case instead.",
                valid_action=False,
            )
            return self._build_step_result(reward, {"valid_action": False, "error": "resolved_case"})

        if action.action_type == ActionTypeEnum.REVIEW_TRANSACTION:
            reward = self._handle_review(case_id, action)
        elif action.action_type == ActionTypeEnum.FETCH_CUSTOMER_PROFILE:
            reward = self._handle_fetch(case_id, action, "customer_profile", CaseScreenEnum.CUSTOMER_PROFILE)
        elif action.action_type == ActionTypeEnum.FETCH_MERCHANT_PROFILE:
            reward = self._handle_fetch(case_id, action, "merchant_profile", CaseScreenEnum.MERCHANT_PROFILE)
        elif action.action_type == ActionTypeEnum.FETCH_NETWORK_GRAPH:
            reward = self._handle_fetch(case_id, action, "network_graph", CaseScreenEnum.CASE_CONSOLE)
        elif action.action_type == ActionTypeEnum.CHECK_POLICY:
            reward = self._handle_fetch(case_id, action, "policy_guide", CaseScreenEnum.POLICY_ESCALATION)
        elif action.action_type == ActionTypeEnum.ADD_CASE_NOTE:
            reward = self._handle_add_note(case_id, action)
        elif action.action_type == ActionTypeEnum.RESOLVE_CASE:
            reward = self._handle_resolve(case_id, action)
        else:  # pragma: no cover - enum already constrains values
            reward = self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.25,
                reason=f"Unsupported action_type '{action.action_type.value}'.",
                valid_action=False,
            )

        return self._build_step_result(
            reward,
            {
                "valid_action": reward.value > -0.999 and not reward.reason.startswith("Unknown case_id"),
                "active_case_id": self.active_case_id,
                "resolved_case_ids": self._resolved_case_ids(),
                "remaining_sla": self._remaining_sla(),
                "remaining_steps": self._remaining_steps(),
            },
        )

    def state(self) -> EpisodeState:
        """Return the current episode state."""

        return EpisodeState(
            episode_id=self.episode_id,
            task_name=self.current_task,
            current_screen=self.current_screen,
            active_case_id=self.active_case_id,
            step_count=self.step_count,
            remaining_steps=self._remaining_steps(),
            remaining_sla=self._remaining_sla(),
            cumulative_reward=round(self.cumulative_reward, 4),
            is_done=self.is_done,
            resolved_case_ids=self._resolved_case_ids(),
            unresolved_case_ids=self._unresolved_case_ids(),
            notes_written_by_case={case_id: data["note_count"] for case_id, data in self.case_state.items()},
            evidence_keys_by_case={
                case_id: sorted(data["revealed_evidence"].keys()) for case_id, data in self.case_state.items()
            },
            policy_checked_case_ids=sorted(
                case_id for case_id, data in self.case_state.items() if data["policy_checked"]
            ),
            resolution_by_case={
                case_id: data["resolution"]
                for case_id, data in self.case_state.items()
                if data["resolution"] is not None
            },
            invalid_action_count=self.invalid_action_count,
            redundant_action_count=self.redundant_action_count,
        )

    def get_episode_report(self) -> Dict[str, Any]:
        """Return a deterministic grading report for the current or completed episode."""

        case_reports = [self._build_case_report(case_id) for case_id in self.case_order]
        case_count = max(1, len(case_reports))
        resolution_accuracy = sum(1.0 if report["resolution_correct"] else 0.0 for report in case_reports) / case_count
        evidence_coverage = sum(report["evidence_coverage"] for report in case_reports) / case_count
        policy_compliance = sum(1.0 if report["policy_compliant"] else 0.0 for report in case_reports) / case_count
        workflow_completion = (
            sum(report["workflow_completion"] for report in case_reports) / case_count if case_reports else 0.0
        )
        overstep_penalty = max(0, self.step_count - TASK_CONFIG[self.current_task]["ideal_steps"]) * 0.05
        efficiency = max(
            0.0,
            1.0
            - (self.invalid_action_count * 0.12)
            - (self.redundant_action_count * 0.08)
            - (self.note_spam_count * 0.06)
            - overstep_penalty,
        )

        if self.current_task == TaskDifficulty.HARD:
            reviewed_count = sum(1.0 if report["reviewed"] else 0.0 for report in case_reports)
            network_count = sum(
                1.0 if "network_graph" in report["revealed_evidence"] else 0.0 for report in case_reports
            )
            resolved_count = sum(1.0 if report["submitted_resolution"] else 0.0 for report in case_reports)
            link_consistency = min(
                1.0,
                0.25 * reviewed_count + 0.25 * network_count + 0.25 * resolved_count + 0.25 * resolution_accuracy * 2,
            )
        else:
            link_consistency = 1.0

        summary = {
            "episode_id": self.episode_id,
            "task": self.current_task.value,
            "step_count": self.step_count,
            "max_steps": TASK_CONFIG[self.current_task]["max_steps"],
            "remaining_sla": self._remaining_sla(),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "invalid_action_count": self.invalid_action_count,
            "redundant_action_count": self.redundant_action_count,
            "note_spam_count": self.note_spam_count,
            "case_summaries": case_reports,
            "metrics": {
                "resolution_accuracy": round(resolution_accuracy, 4),
                "evidence_coverage": round(evidence_coverage, 4),
                "policy_compliance": round(policy_compliance, 4),
                "workflow_completion": round(workflow_completion, 4),
                "efficiency": round(efficiency, 4),
                "link_consistency": round(link_consistency, 4),
            },
            "audit_log": list(self.audit_log),
        }
        self.last_episode_summary = summary
        return summary

    def _build_workflow_cases(self, task: TaskDifficulty) -> Dict[str, Dict[str, Any]]:
        if task == TaskDifficulty.EASY:
            source_case = self._select_easy_case(self.data_loader.get_task_cases("easy"))
            return {
                "easy_case_01": self._make_workflow_case(
                    case_id="easy_case_01",
                    raw_case=source_case,
                    queue_reason="High-value purchase queued for a quick manual review.",
                    correct_resolution=ResolutionEnum.BLOCK,
                    required_tools={"transaction_review", "case_note"},
                    useful_tools={"merchant_profile"},
                    policy_required=False,
                    linked_case_ids=[],
                    role="single",
                )
            }

        if task == TaskDifficulty.MEDIUM:
            source_case = self._select_medium_case(self.data_loader.get_task_cases("medium"))
            return {
                "medium_case_01": self._make_workflow_case(
                    case_id="medium_case_01",
                    raw_case=source_case,
                    queue_reason="Mixed signals triggered review and supporting evidence is needed.",
                    correct_resolution=ResolutionEnum.REQUEST_DOCS,
                    required_tools={"transaction_review", "customer_profile", "policy_guide", "case_note"},
                    useful_tools={"merchant_profile"},
                    policy_required=True,
                    linked_case_ids=[],
                    role="single",
                )
            }

        hard_cases = self._select_hard_pair(self.data_loader.get_task_cases("hard"))
        primary = self._make_workflow_case(
            case_id="hard_case_primary",
            raw_case=hard_cases[0],
            queue_reason="Operational anomaly spike triggered a higher-touch review.",
            correct_resolution=ResolutionEnum.ESCALATE,
            required_tools={"transaction_review", "network_graph", "merchant_profile", "policy_guide", "case_note"},
            useful_tools=set(),
            policy_required=True,
            linked_case_ids=["hard_case_secondary"],
            role="primary",
        )
        secondary = self._make_workflow_case(
            case_id="hard_case_secondary",
            raw_case=hard_cases[1],
            queue_reason="A related anomaly surfaced in the same review wave.",
            correct_resolution=ResolutionEnum.BLOCK,
            required_tools={"transaction_review", "network_graph", "customer_profile", "policy_guide", "case_note"},
            useful_tools=set(),
            policy_required=True,
            linked_case_ids=["hard_case_primary"],
            role="secondary",
        )
        return {primary["case_id"]: primary, secondary["case_id"]: secondary}

    def _select_easy_case(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        fraud_cases = [case for case in cases if case["label"] == "fraud"]
        return max(fraud_cases, key=lambda case: (case["risk_score"], case["business_cost"]))

    def _select_medium_case(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidates = [
            case
            for case in cases
            if case["label"] == "legitimate"
            and case["transaction_data"]["previous_fraud_flags"] >= 1
            and case["transaction_data"]["seller_chargeback_rate_30d"] >= 0.04
        ]
        if not candidates:
            candidates = [case for case in cases if case["label"] == "legitimate"]
        return max(candidates, key=lambda case: (case["risk_score"], case["business_cost"]))

    def _select_hard_pair(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for case in cases:
            seller_id = case["transaction_data"]["seller_id"]
            groups.setdefault(seller_id, []).append(case)

        linked_groups = [
            group
            for group in groups.values()
            if len(group) >= 2 and all(case["label"] == "fraud" for case in group)
        ]
        if not linked_groups:
            linked_groups = [cases[:2]]

        chosen_group = max(linked_groups, key=lambda group: sum(case["business_cost"] for case in group))
        ordered = sorted(chosen_group, key=lambda case: (case["business_cost"], case["risk_score"]), reverse=True)
        return ordered[:2]

    def _make_workflow_case(
        self,
        case_id: str,
        raw_case: Dict[str, Any],
        queue_reason: str,
        correct_resolution: ResolutionEnum,
        required_tools: set[str],
        useful_tools: set[str],
        policy_required: bool,
        linked_case_ids: List[str],
        role: str,
    ) -> Dict[str, Any]:
        transaction = copy.deepcopy(raw_case["transaction_data"])
        history = copy.deepcopy(raw_case["historical_context"])
        risk_score = float(raw_case["risk_score"])
        business_cost = float(raw_case["business_cost"])
        shipping_country = transaction["shipping_address"]
        device_country = transaction["device_country"]
        geo_mismatch = shipping_country != device_country
        timestamp = transaction["timestamp"]

        queue_card = {
            "case_id": case_id,
            "priority": self._priority_label(risk_score, business_cost),
            "queue_reason": queue_reason,
            "visible_risk_band": "review",
            "status": "triage",
            "linked_case_ids": [],
        }

        transaction_review = {
            "view": CaseScreenEnum.CASE_CONSOLE.value,
            "summary": self._transaction_summary(transaction, geo_mismatch),
            "facts": {
                "amount_usd": transaction["amount"],
                "item_category": transaction["item_category"],
                "timestamp": timestamp,
                "shipping_country": shipping_country,
                "device_country": device_country,
                "payment_method": transaction["payment_method"],
                "shipping_speed": transaction["shipping_speed"],
                "same_address_orders_24h": transaction["same_address_orders_24h"],
            },
        }
        customer_profile = {
            "view": CaseScreenEnum.CUSTOMER_PROFILE.value,
            "summary": self._customer_summary(transaction),
            "facts": {
                "buyer_account_age_days": transaction["buyer_account_age_days"],
                "buyer_disputes_90d": transaction["buyer_disputes_90d"],
                "is_repeat_buyer": transaction["is_repeat_buyer"],
            },
        }
        merchant_profile = {
            "view": CaseScreenEnum.MERCHANT_PROFILE.value,
            "summary": self._merchant_summary(transaction),
            "facts": {
                "seller_account_age_days": transaction["seller_account_age_days"],
                "seller_avg_rating": transaction["seller_avg_rating"],
                "num_seller_reviews": transaction["num_seller_reviews"],
                "seller_chargeback_rate_30d": transaction["seller_chargeback_rate_30d"],
            },
        }
        network_graph = {
            "view": CaseScreenEnum.CASE_CONSOLE.value,
            "summary": self._network_summary(role, linked_case_ids, history),
            "facts": {
                "shared_device_accounts_24h": transaction["shared_device_accounts_24h"],
                "previous_fraud_flags": transaction["previous_fraud_flags"],
                "cluster_alert_score": history["cluster_alert_score"],
                "linked_cards_7d": history["linked_cards_7d"],
                "linked_case_ids": list(linked_case_ids),
            },
        }
        policy_guide = {
            "view": CaseScreenEnum.POLICY_ESCALATION.value,
            "summary": self._policy_summary(policy_required, business_cost),
            "facts": {
                "policy_required": policy_required,
                "note_required": True,
                "request_docs_on_unresolved_conflict": self.current_task != TaskDifficulty.EASY,
                "escalate_if_cluster_and_loss": role == "primary" and business_cost >= 1.35,
                "high_loss_threshold": 1.35,
            },
        }

        return {
            "case_id": case_id,
            "raw_case": raw_case,
            "transaction": transaction,
            "history": history,
            "risk_score": risk_score,
            "business_cost": business_cost,
            "correct_resolution": correct_resolution,
            "required_tools": set(required_tools),
            "useful_tools": set(useful_tools),
            "policy_required": policy_required,
            "linked_case_ids": list(linked_case_ids),
            "role": role,
            "queue_card": queue_card,
            "hidden_flags": {
                "payment_ops_high_risk": self._high_risk_payment_ops(transaction, geo_mismatch),
                "network_high_risk": history["cluster_alert_score"] >= 0.7,
            },
            "evidence_catalog": {
                "transaction_review": transaction_review,
                "customer_profile": customer_profile,
                "merchant_profile": merchant_profile,
                "network_graph": network_graph,
                "policy_guide": policy_guide,
            },
        }

    def _priority_label(self, risk_score: float, business_cost: float) -> str:
        if risk_score >= 0.68 or business_cost >= 1.45:
            return "P1"
        if risk_score >= 0.5 or business_cost >= 1.1:
            return "P2"
        return "P3"

    def _transaction_summary(self, transaction: Dict[str, Any], geo_mismatch: bool) -> str:
        return (
            f"Payment {transaction['payment_method']}; shipping {transaction['shipping_speed']}; "
            f"same-address orders={transaction['same_address_orders_24h']}; geo mismatch={geo_mismatch}."
        )

    def _customer_summary(self, transaction: Dict[str, Any]) -> str:
        return (
            f"Buyer age {transaction['buyer_account_age_days']}d; disputes {transaction['buyer_disputes_90d']}; "
            f"repeat buyer={transaction['is_repeat_buyer']}."
        )

    def _merchant_summary(self, transaction: Dict[str, Any]) -> str:
        return (
            f"Seller rating {transaction['seller_avg_rating']:.2f}; reviews {transaction['num_seller_reviews']}; "
            f"chargeback rate {transaction['seller_chargeback_rate_30d']:.3f}."
        )

    def _network_summary(self, role: str, linked_case_ids: List[str], history: Dict[str, Any]) -> str:
        if not linked_case_ids:
            return (
                f"Graph review surfaced cluster score {history['cluster_alert_score']:.2f} "
                "with no immediately visible linked cases."
            )
        if role == "primary":
            return (
                f"Graph review surfaced a cluster score of {history['cluster_alert_score']:.2f} "
                "and a shared-entity pattern worth escalation review."
            )
        return (
            f"Graph review surfaced a cluster score of {history['cluster_alert_score']:.2f} "
            "and a related-activity pattern on this case."
        )

    def _policy_summary(self, policy_required: bool, business_cost: float) -> str:
        if not policy_required:
            return "Policy allows direct approve or block decisions once a note is added."
        if business_cost >= 1.35:
            return "Policy recommends escalation when hidden network risk and business impact are both elevated."
        return "Policy recommends requesting documents or holding the case when signals remain mixed."

    def _high_risk_payment_ops(self, transaction: Dict[str, Any], geo_mismatch: bool) -> bool:
        return bool(
            transaction["payment_method"] in {"prepaid_card", "gift_card", "crypto_gateway"}
            or transaction["shipping_speed"] in {"same-day", "overnight"}
            or transaction["same_address_orders_24h"] >= 5
            or geo_mismatch
        )

    def _handle_review(self, case_id: str, action: FraudCheckAction) -> Reward:
        self.active_case_id = case_id
        self.current_screen = CaseScreenEnum.CASE_CONSOLE
        state = self.case_state[case_id]
        case = self.workflow_cases[case_id]
        if state["reviewed"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.05,
                reason="Transaction review was already completed for this case.",
                evidence_key="transaction_review",
                redundant=True,
            )

        state["reviewed"] = True
        state["status"] = "in_review"
        state["revealed_evidence"]["transaction_review"] = case["evidence_catalog"]["transaction_review"]
        bonus = 0.08 if case["hidden_flags"]["payment_ops_high_risk"] else 0.0
        reason = "Transaction review revealed the operational transaction trace."
        if bonus > 0:
            reason += " The review surfaced high-risk payment or fulfillment signals."
        return self._apply_action_outcome(
            action=action,
            case_id=case_id,
            base_value=0.04 + bonus,
            reason=reason,
            evidence_key="transaction_review",
        )

    def _handle_fetch(
        self,
        case_id: str,
        action: FraudCheckAction,
        evidence_key: str,
        screen: CaseScreenEnum,
    ) -> Reward:
        self.active_case_id = case_id
        state = self.case_state[case_id]
        case = self.workflow_cases[case_id]

        if not state["reviewed"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.14,
                reason="Open the transaction review before pulling deeper evidence.",
                valid_action=False,
            )

        self.current_screen = screen
        if evidence_key in state["revealed_evidence"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.05,
                reason=f"{evidence_key} was already fetched for this case.",
                evidence_key=evidence_key,
                redundant=True,
            )

        over_budget = state["fetch_budget_remaining"] <= 0
        if not over_budget:
            state["fetch_budget_remaining"] -= 1
        state["fetches_used"] += 1
        state["revealed_evidence"][evidence_key] = case["evidence_catalog"][evidence_key]
        state["status"] = "investigating"
        if evidence_key == "policy_guide":
            state["policy_checked"] = True

        useful = evidence_key in case["required_tools"] or evidence_key in case["useful_tools"]
        base_value = 0.05 if useful else 0.0
        reason = f"{evidence_key} revealed new hidden evidence."
        if evidence_key == "network_graph" and case["hidden_flags"]["network_high_risk"]:
            base_value += 0.08
            reason = "Network graph revealed high-risk cluster evidence before the final decision."
        if over_budget:
            base_value -= 0.03
            reason += " The fetch happened after the investigation budget was exhausted."

        return self._apply_action_outcome(
            action=action,
            case_id=case_id,
            base_value=base_value,
            reason=reason,
            evidence_key=evidence_key,
        )

    def _handle_add_note(self, case_id: str, action: FraudCheckAction) -> Reward:
        self.active_case_id = case_id
        self.current_screen = CaseScreenEnum.CASE_CONSOLE
        state = self.case_state[case_id]

        if not state["reviewed"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.16,
                reason="Notes are only allowed after the transaction has been reviewed.",
                valid_action=False,
            )

        assert action.note_text is not None  # validated by Pydantic
        normalized_note = action.note_text.strip().lower()
        existing_notes = [note.lower() for note in state["notes"]]
        if normalized_note in existing_notes or len(existing_notes) >= 2:
            self.note_spam_count += 1
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.12,
                reason="Repeated or low-value note spam is penalized.",
                anti_hacking=True,
                redundant=True,
            )

        state["notes"].append(action.note_text.strip())
        state["note_count"] += 1
        state["status"] = "documented"
        return self._apply_action_outcome(
            action=action,
            case_id=case_id,
            base_value=0.09,
            reason="Added a case note that documents the investigation state.",
            evidence_key="case_note",
        )

    def _handle_resolve(self, case_id: str, action: FraudCheckAction) -> Reward:
        self.active_case_id = case_id
        self.current_screen = CaseScreenEnum.POLICY_ESCALATION
        state = self.case_state[case_id]
        case = self.workflow_cases[case_id]

        if not state["reviewed"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.35,
                reason="Cannot resolve a case before reviewing the transaction.",
                valid_action=False,
            )

        assert action.resolution is not None  # validated by Pydantic

        missing_required = [
            tool for tool in case["required_tools"] if tool not in self._completed_tool_markers(case_id)
        ]
        note_missing = state["note_count"] == 0
        policy_missing = case["policy_required"] and not state["policy_checked"]
        no_fetch_evidence = (
            self.current_task in {TaskDifficulty.MEDIUM, TaskDifficulty.HARD} and state["fetches_used"] == 0
        )
        used_investigation_bonus = (
            self.current_task in {TaskDifficulty.MEDIUM, TaskDifficulty.HARD}
            and state["fetches_used"] >= TASK_CONFIG[self.current_task]["minimum_fetches_for_bonus"]
        )
        correct = action.resolution == case["correct_resolution"]
        policy_compliant = correct and (not policy_missing)

        base_value = 0.72 if correct and not missing_required and not note_missing else 0.38 if correct else -0.72
        if policy_missing and correct:
            base_value -= 0.28
        if note_missing:
            base_value -= 0.20
        if missing_required and correct:
            base_value -= 0.16 * min(2, len(missing_required))
        if no_fetch_evidence:
            base_value -= 0.10
        if correct and used_investigation_bonus:
            base_value += 0.15

        reason_parts = []
        if correct:
            reason_parts.append("Resolution matched the hidden correct routing.")
        else:
            reason_parts.append(
                f"Incorrect routing: expected {case['correct_resolution'].value}, got {action.resolution.value}."
            )
        if policy_missing:
            reason_parts.append("Policy was not checked before resolution.")
        if note_missing:
            reason_parts.append("A case note was required before closure.")
        if missing_required:
            reason_parts.append(f"Missing required workflow steps: {', '.join(sorted(missing_required))}.")
        if no_fetch_evidence:
            reason_parts.append("Medium and hard cases require at least one investigation fetch before resolution.")
        if correct and used_investigation_bonus:
            reason_parts.append("The route also earned the investigation-use bonus.")

        state["resolution"] = action.resolution
        state["resolved"] = True
        state["resolution_correct"] = correct
        state["policy_compliant"] = policy_compliant
        state["status"] = "resolved"

        unresolved = self._unresolved_case_ids()
        if not unresolved:
            self.current_screen = CaseScreenEnum.QUEUE
        else:
            self.active_case_id = unresolved[0]
            self.current_screen = CaseScreenEnum.QUEUE

        return self._apply_action_outcome(
            action=action,
            case_id=case_id,
            base_value=base_value,
            reason=" ".join(reason_parts),
            resolution=action.resolution,
            ground_truth_resolution=case["correct_resolution"],
            is_correct=correct,
            policy_compliant=policy_compliant,
        )

    def _completed_tool_markers(self, case_id: str) -> set[str]:
        state = self.case_state[case_id]
        completed = set(state["revealed_evidence"].keys())
        if state["note_count"] > 0:
            completed.add("case_note")
        return completed

    def _apply_action_outcome(
        self,
        action: FraudCheckAction,
        case_id: str,
        base_value: float,
        reason: str,
        evidence_key: str | None = None,
        resolution: ResolutionEnum | None = None,
        ground_truth_resolution: ResolutionEnum | None = None,
        is_correct: bool | None = None,
        policy_compliant: bool | None = None,
        valid_action: bool = True,
        redundant: bool = False,
        anti_hacking: bool = False,
    ) -> Reward:
        state = self.case_state.get(case_id)
        if state is not None:
            state["action_history"].append(action.action_type.value)
            if not valid_action:
                state["invalid_actions"] += 1
                self.invalid_action_count += 1
            if redundant:
                state["redundant_actions"] += 1
                self.redundant_action_count += 1
            if anti_hacking:
                state["anti_hacking_hits"] += 1

        action_cost = 0.02 if action.action_type in FETCH_ACTIONS else 0.0
        if action.action_type == ActionTypeEnum.ADD_CASE_NOTE:
            action_cost = 0.01

        projected_step = self.step_count + 1
        sla_penalty = 0.06 * max(0, projected_step - self.sla_limit[self.current_task])
        reward_value = max(-1.0, min(1.0, base_value - action_cost - sla_penalty))

        reward = Reward(
            value=round(reward_value, 4),
            reason=reason,
            action_type=action.action_type,
            case_id=case_id,
            action_cost=round(action_cost, 4),
            sla_penalty=round(sla_penalty, 4),
            evidence_key=evidence_key,
            resolution=resolution,
            ground_truth_resolution=ground_truth_resolution,
            is_correct=is_correct,
            policy_compliant=policy_compliant,
            anti_hacking_triggered=anti_hacking,
        )

        self.step_count = projected_step
        self.cumulative_reward += reward.value
        if self.step_count >= self.max_steps[self.current_task]:
            self.is_done = True
        if not self._unresolved_case_ids():
            self.is_done = True

        self.audit_log.append(
            {
                "step": self.step_count,
                "case_id": case_id,
                "action_type": action.action_type.value,
                "reward": reward.value,
                "reason": reason,
                "screen": self.current_screen.value,
                "resolved_case_ids": self._resolved_case_ids(),
            }
        )
        return reward

    def _build_step_result(self, reward: Reward, extra_info: Dict[str, Any]) -> StepResult:
        observation = self._build_observation()
        if self.is_done:
            self.last_episode_summary = self.get_episode_report()
        info = {
            "episode_id": self.episode_id,
            "task": self.current_task.value,
            **extra_info,
        }
        return StepResult(observation=observation, reward=reward, done=self.is_done, info=info)

    def _build_observation(self) -> FraudCheckObservation:
        case_id = self.active_case_id or self.case_order[0]
        case = self.workflow_cases[case_id]
        state = self.case_state[case_id]
        is_triage_only = not state["reviewed"] and self.step_count == 0

        case_summary = CaseSummary(
            case_id=case_id,
            status=state["status"],
            queue_reason=case["queue_card"]["queue_reason"],
            visible_risk_band="review",
            amount_usd=float(case["transaction"]["amount"]),
            merchant_region="masked" if not state["reviewed"] else case["transaction"]["shipping_address"],
            evidence_collected=sorted(state["revealed_evidence"].keys()),
            note_added=state["note_count"] > 0,
        )

        visible_panels = ["triage_summary"] if is_triage_only else ["triage_summary", "evidence_panel"]
        if state["reviewed"]:
            visible_panels.append(self.current_screen.value.lower().replace(" ", "_"))
        visible_panels.extend(sorted(state["revealed_evidence"].keys()))
        if state["note_count"] > 0:
            visible_panels.append("case_notes")

        linked_case_ids = []
        if "network_graph" in state["revealed_evidence"]:
            linked_case_ids = list(case["linked_case_ids"])

        queue_items: List[QueueCaseCard] = []
        if state["reviewed"]:
            queue_items = [
                QueueCaseCard(
                    case_id=workflow_case["case_id"],
                    priority=workflow_case["queue_card"]["priority"],
                    queue_reason=workflow_case["queue_card"]["queue_reason"],
                    visible_risk_band="review",
                    status=self.case_state[workflow_case["case_id"]]["status"],
                    linked_case_ids=[],
                )
                for workflow_case in self.workflow_cases.values()
            ]

        return FraudCheckObservation(
            case_id=case_id,
            task_name=self.current_task,
            current_screen=self.current_screen,
            visible_panels=visible_panels,
            revealed_evidence=copy.deepcopy(state["revealed_evidence"]),
            linked_case_ids=linked_case_ids,
            remaining_steps=self._remaining_steps(),
            remaining_sla=self._remaining_sla(),
            note_required=state["note_count"] == 0,
            allowed_actions=self._allowed_actions(case_id),
            queue_items=queue_items,
            case_summary=case_summary,
            episode_step=self.step_count,
            app_context={
                "item_category": case["transaction"]["item_category"],
                "timestamp": case["transaction"]["timestamp"],
                "investigation_budget_remaining": state["fetch_budget_remaining"],
                "available_investigations": sorted(action.value for action in FETCH_ACTIONS),
                "task_description": TASK_CONFIG[self.current_task]["description"],
            },
        )

    def _allowed_actions(self, case_id: str) -> List[ActionTypeEnum]:
        if self.is_done:
            return []

        state = self.case_state[case_id]
        if state["resolved"]:
            return [ActionTypeEnum.REVIEW_TRANSACTION] if self._unresolved_case_ids() else []

        if not state["reviewed"]:
            return [ActionTypeEnum.REVIEW_TRANSACTION]

        return [
            ActionTypeEnum.REVIEW_TRANSACTION,
            ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
            ActionTypeEnum.FETCH_MERCHANT_PROFILE,
            ActionTypeEnum.FETCH_NETWORK_GRAPH,
            ActionTypeEnum.CHECK_POLICY,
            ActionTypeEnum.ADD_CASE_NOTE,
            ActionTypeEnum.RESOLVE_CASE,
        ]

    def _remaining_steps(self) -> int:
        return max(0, self.max_steps[self.current_task] - self.step_count)

    def _remaining_sla(self) -> int:
        return max(0, self.sla_limit[self.current_task] - self.step_count)

    def _resolved_case_ids(self) -> List[str]:
        return [case_id for case_id in self.case_order if self.case_state[case_id]["resolved"]]

    def _unresolved_case_ids(self) -> List[str]:
        return [case_id for case_id in self.case_order if not self.case_state[case_id]["resolved"]]

    def _build_case_report(self, case_id: str) -> Dict[str, Any]:
        case = self.workflow_cases[case_id]
        state = self.case_state[case_id]
        required_tools = case["required_tools"]
        completed_tools = self._completed_tool_markers(case_id)
        coverage = len(required_tools & completed_tools) / max(1, len(required_tools))
        workflow_completion = (
            (1.0 if state["reviewed"] else 0.0)
            + (1.0 if state["note_count"] > 0 else 0.0)
            + (1.0 if state["resolved"] else 0.0)
            + coverage
        ) / 4.0
        return {
            "case_id": case_id,
            "queue_reason": case["queue_card"]["queue_reason"],
            "correct_resolution": case["correct_resolution"].value,
            "submitted_resolution": state["resolution"].value if state["resolution"] else None,
            "reviewed": state["reviewed"],
            "note_count": state["note_count"],
            "policy_checked": state["policy_checked"],
            "revealed_evidence": sorted(state["revealed_evidence"].keys()),
            "resolution_correct": state["resolution_correct"],
            "policy_compliant": state["policy_compliant"],
            "invalid_actions": state["invalid_actions"],
            "redundant_actions": state["redundant_actions"],
            "anti_hacking_hits": state["anti_hacking_hits"],
            "linked_case_ids": list(case["linked_case_ids"]),
            "evidence_coverage": round(coverage, 4),
            "workflow_completion": round(workflow_completion, 4),
        }
