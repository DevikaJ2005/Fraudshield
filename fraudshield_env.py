"""FraudShield enterprise FraudOps environment implementation."""

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
        "description": "Single low-noise case with an obvious routing decision.",
    },
    TaskDifficulty.MEDIUM: {
        "source_task": "medium",
        "num_cases": 1,
        "max_steps": 8,
        "sla_limit": 6,
        "ideal_steps": 5,
        "description": "Single ambiguous case that requires profile review and policy lookup.",
    },
    TaskDifficulty.HARD: {
        "source_task": "hard",
        "num_cases": 2,
        "max_steps": 14,
        "sla_limit": 11,
        "ideal_steps": 9,
        "description": "Two linked fraud cases that require network reasoning and escalation policy.",
    },
}


class FraudShieldEnvironment:
    """OpenEnv-compatible enterprise fraud-operations environment."""

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
        """Start a new enterprise workflow episode."""

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
                "status": "queued",
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
            }
            for case_id in self.case_order
        }

        info = {
            "episode_id": self.episode_id,
            "task": self.current_task.value,
            "num_cases": config["num_cases"],
            "max_steps": config["max_steps"],
            "sla_limit": config["sla_limit"],
            "description": config["description"],
            "apps": [screen.value for screen in CaseScreenEnum],
            "data_snapshot": self.data_loader.get_bundle_summary(),
        }
        return ResetResult(observation=self._build_observation(), info=info)

    def step(self, action: FraudCheckAction) -> StepResult:
        """Apply a single workflow action."""

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
        case = self.workflow_cases[case_id]

        if state["resolved"] and action.action_type != ActionTypeEnum.REVIEW_TRANSACTION:
            reward = self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.22,
                reason="Resolved cases cannot be modified; switch back to the queue or another open case.",
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
            sum(report["workflow_completion"] for report in case_reports) / case_count
            if case_reports
            else 0.0
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
                    queue_reason="New seller plus geo mismatch triggered a manual review queue.",
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
                    queue_reason="Conflicting customer and merchant signals require policy-aware review.",
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
            queue_reason="Linked merchant cluster with repeated abuse indicators needs escalation review.",
            correct_resolution=ResolutionEnum.ESCALATE,
            required_tools={"transaction_review", "network_graph", "policy_guide", "case_note"},
            useful_tools={"merchant_profile"},
            policy_required=True,
            linked_case_ids=["hard_case_secondary"],
            role="primary",
        )
        secondary = self._make_workflow_case(
            case_id="hard_case_secondary",
            raw_case=hard_cases[1],
            queue_reason="Connected follow-on case shares entities with an open fraud ring alert.",
            correct_resolution=ResolutionEnum.BLOCK,
            required_tools={"transaction_review", "network_graph", "policy_guide", "case_note"},
            useful_tools={"customer_profile"},
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

        queue_card = {
            "case_id": case_id,
            "priority": self._priority_label(risk_score, business_cost),
            "queue_reason": queue_reason,
            "visible_risk_band": self._risk_band(risk_score),
            "status": "queued",
            "linked_case_ids": list(linked_case_ids),
        }

        transaction_review = {
            "app": CaseScreenEnum.CASE_CONSOLE.value,
            "summary": self._transaction_summary(transaction, risk_score, geo_mismatch),
            "facts": {
                "amount_usd": transaction["amount"],
                "item_category": transaction["item_category"],
                "shipping_country": shipping_country,
                "device_country": device_country,
                "payment_method": transaction["payment_method"],
                "seller_account_age_days": transaction["seller_account_age_days"],
                "previous_fraud_flags": transaction["previous_fraud_flags"],
                "shared_device_accounts_24h": transaction["shared_device_accounts_24h"],
                "same_address_orders_24h": transaction["same_address_orders_24h"],
            },
            "alerts": self._transaction_alerts(transaction, history),
        }
        customer_profile = {
            "app": CaseScreenEnum.CUSTOMER_PROFILE.value,
            "summary": self._customer_summary(transaction, history),
            "facts": {
                "buyer_account_age_days": transaction["buyer_account_age_days"],
                "buyer_disputes_90d": transaction["buyer_disputes_90d"],
                "is_repeat_buyer": transaction["is_repeat_buyer"],
                "linked_cards_7d": history["linked_cards_7d"],
                "recent_refunds_7d": history["recent_refunds_7d"],
            },
        }
        merchant_profile = {
            "app": CaseScreenEnum.MERCHANT_PROFILE.value,
            "summary": self._merchant_summary(transaction, history),
            "facts": {
                "seller_account_age_days": transaction["seller_account_age_days"],
                "seller_avg_rating": transaction["seller_avg_rating"],
                "num_seller_reviews": transaction["num_seller_reviews"],
                "seller_chargeback_rate_30d": transaction["seller_chargeback_rate_30d"],
                "seller_transactions_1h": history["seller_transactions_1h"],
            },
        }
        network_graph = {
            "app": CaseScreenEnum.CASE_CONSOLE.value,
            "summary": self._network_summary(role, linked_case_ids, history),
            "facts": {
                "cluster_alert_score": history["cluster_alert_score"],
                "linked_case_ids": list(linked_case_ids),
                "shared_seller_id": transaction["seller_id"],
                "shared_buyer_pattern": transaction["buyer_id"].startswith("buyer_linked"),
            },
        }
        policy_guide = {
            "app": CaseScreenEnum.POLICY_ESCALATION.value,
            "summary": self._policy_summary(correct_resolution, policy_required, role),
            "facts": {
                "policy_required": policy_required,
                "correct_resolution_hint": correct_resolution.value,
                "escalate_if_business_cost_above": 1.35,
                "request_docs_if_flags_and_chargebacks": True,
                "review_note_required": True,
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

    def _risk_band(self, risk_score: float) -> str:
        if risk_score >= 0.68:
            return "high"
        if risk_score >= 0.45:
            return "medium"
        return "low"

    def _transaction_summary(self, transaction: Dict[str, Any], risk_score: float, geo_mismatch: bool) -> str:
        seller_age = transaction["seller_account_age_days"]
        flags = transaction["previous_fraud_flags"]
        return (
            f"Amount ${transaction['amount']:.2f}; risk band {self._risk_band(risk_score)}; "
            f"seller age {seller_age}d; prior flags {flags}; geo mismatch={geo_mismatch}."
        )

    def _transaction_alerts(self, transaction: Dict[str, Any], history: Dict[str, Any]) -> List[str]:
        alerts: List[str] = []
        if transaction["seller_account_age_days"] <= 90:
            alerts.append("Seller account is newly created.")
        if transaction["device_country"] != transaction["shipping_address"]:
            alerts.append("Device country does not match shipping country.")
        if transaction["shared_device_accounts_24h"] >= 6:
            alerts.append("Device was reused across multiple accounts in the last 24h.")
        if history["cluster_alert_score"] >= 0.7:
            alerts.append("Cluster alert score is elevated.")
        if not alerts:
            alerts.append("No single transaction alert is decisive on its own.")
        return alerts

    def _customer_summary(self, transaction: Dict[str, Any], history: Dict[str, Any]) -> str:
        return (
            f"Buyer age {transaction['buyer_account_age_days']}d; disputes {transaction['buyer_disputes_90d']}; "
            f"repeat buyer={transaction['is_repeat_buyer']}; linked cards {history['linked_cards_7d']}."
        )

    def _merchant_summary(self, transaction: Dict[str, Any], history: Dict[str, Any]) -> str:
        return (
            f"Seller rating {transaction['seller_avg_rating']:.2f}; reviews {transaction['num_seller_reviews']}; "
            f"chargeback rate {transaction['seller_chargeback_rate_30d']:.3f}; velocity {history['seller_transactions_1h']}/h."
        )

    def _network_summary(self, role: str, linked_case_ids: List[str], history: Dict[str, Any]) -> str:
        if not linked_case_ids:
            return "No strong linked-case pattern is visible from the current graph sample."
        if role == "primary":
            return (
                f"Graph shows repeated shared entities across {len(linked_case_ids) + 1} fraud alerts; "
                f"cluster score {history['cluster_alert_score']:.2f} suggests coordinated activity."
            )
        return (
            f"Graph ties this case to {', '.join(linked_case_ids)} through shared seller/device patterns; "
            f"cluster score {history['cluster_alert_score']:.2f}."
        )

    def _policy_summary(self, correct_resolution: ResolutionEnum, policy_required: bool, role: str) -> str:
        if not policy_required:
            return "Policy allows direct approve/block resolution after a valid note when evidence is clear."
        if role == "primary" and correct_resolution == ResolutionEnum.ESCALATE:
            return "Escalate high-loss linked clusters when ring evidence and business impact are both high."
        if correct_resolution == ResolutionEnum.REQUEST_DOCS:
            return "Request documents when mixed signals remain after profile review."
        return "Use policy routing before final resolution; high-risk linked cases should not be approved."

    def _handle_review(self, case_id: str, action: FraudCheckAction) -> Reward:
        self.active_case_id = case_id
        self.current_screen = CaseScreenEnum.CASE_CONSOLE
        state = self.case_state[case_id]
        if state["reviewed"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.08,
                reason="Transaction review was already opened for this case.",
                evidence_key="transaction_review",
                redundant=True,
            )

        state["reviewed"] = True
        state["status"] = "in_review"
        state["revealed_evidence"]["transaction_review"] = self.workflow_cases[case_id]["evidence_catalog"][
            "transaction_review"
        ]
        return self._apply_action_outcome(
            action=action,
            case_id=case_id,
            base_value=0.09,
            reason="Transaction review opened the Case Console and revealed the core case facts.",
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
                reason="Open the transaction in Case Console before pulling deeper evidence.",
                valid_action=False,
            )

        self.current_screen = screen
        if evidence_key in state["revealed_evidence"]:
            return self._apply_action_outcome(
                action=action,
                case_id=case_id,
                base_value=-0.09,
                reason=f"{evidence_key} was already fetched for this case.",
                evidence_key=evidence_key,
                redundant=True,
            )

        state["revealed_evidence"][evidence_key] = case["evidence_catalog"][evidence_key]
        if evidence_key == "policy_guide":
            state["policy_checked"] = True

        useful = evidence_key in case["required_tools"] or evidence_key in case["useful_tools"]
        base_value = 0.08 if useful else 0.03
        reason = f"{evidence_key} added new evidence in {screen.value}."
        if evidence_key == "policy_guide" and case["policy_required"]:
            base_value = 0.10
            reason = "Policy lookup revealed the routing rules required for this case."
        elif evidence_key == "network_graph" and case["linked_case_ids"]:
            base_value = 0.10
            reason = "Network graph connected the linked cases and exposed shared-entity risk."

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
            reason="Added a case note that documents the current investigation state.",
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
            tool
            for tool in case["required_tools"]
            if tool not in self._completed_tool_markers(case_id)
        ]
        note_missing = state["note_count"] == 0
        policy_missing = case["policy_required"] and not state["policy_checked"]
        correct = action.resolution == case["correct_resolution"]
        policy_compliant = correct and (not policy_missing)

        base_value = 0.72 if correct and not missing_required and not note_missing else 0.38 if correct else -0.72
        if policy_missing and correct:
            base_value -= 0.28
        if note_missing:
            base_value -= 0.20
        if missing_required and correct:
            base_value -= 0.16 * min(2, len(missing_required))

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

        state["resolution"] = action.resolution
        state["resolved"] = True
        state["resolution_correct"] = correct
        state["policy_compliant"] = policy_compliant
        state["status"] = "resolved"

        if not self._unresolved_case_ids():
            self.current_screen = CaseScreenEnum.QUEUE
        else:
            self.active_case_id = self._unresolved_case_ids()[0]
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

        action_cost = 0.02 if action.action_type in {
            ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
            ActionTypeEnum.FETCH_MERCHANT_PROFILE,
            ActionTypeEnum.FETCH_NETWORK_GRAPH,
            ActionTypeEnum.CHECK_POLICY,
        } else 0.0
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
        queue_items = [
            QueueCaseCard(
                case_id=workflow_case["case_id"],
                priority=workflow_case["queue_card"]["priority"],
                queue_reason=workflow_case["queue_card"]["queue_reason"],
                visible_risk_band=workflow_case["queue_card"]["visible_risk_band"],
                status=self.case_state[workflow_case["case_id"]]["status"],
                linked_case_ids=workflow_case["linked_case_ids"],
            )
            for workflow_case in self.workflow_cases.values()
        ]
        case_summary = CaseSummary(
            case_id=case_id,
            status=state["status"],
            queue_reason=case["queue_card"]["queue_reason"],
            visible_risk_band=case["queue_card"]["visible_risk_band"],
            amount_usd=float(case["transaction"]["amount"]),
            merchant_region=case["transaction"]["shipping_address"],
            evidence_collected=sorted(state["revealed_evidence"].keys()),
            note_added=state["note_count"] > 0,
        )

        visible_panels = ["queue_table", "sla_banner"]
        if self.current_screen == CaseScreenEnum.CASE_CONSOLE:
            visible_panels.append("case_console")
        elif self.current_screen == CaseScreenEnum.CUSTOMER_PROFILE:
            visible_panels.append("customer_profile")
        elif self.current_screen == CaseScreenEnum.MERCHANT_PROFILE:
            visible_panels.append("merchant_profile")
        elif self.current_screen == CaseScreenEnum.POLICY_ESCALATION:
            visible_panels.append("policy_escalation")
        visible_panels.extend(sorted(state["revealed_evidence"].keys()))
        if state["note_count"] > 0:
            visible_panels.append("case_notes")

        return FraudCheckObservation(
            case_id=case_id,
            task_name=self.current_task,
            current_screen=self.current_screen,
            visible_panels=visible_panels,
            revealed_evidence=copy.deepcopy(state["revealed_evidence"]),
            linked_case_ids=list(case["linked_case_ids"]),
            remaining_steps=self._remaining_steps(),
            remaining_sla=self._remaining_sla(),
            note_required=state["note_count"] == 0,
            allowed_actions=self._allowed_actions(case_id),
            queue_items=queue_items,
            case_summary=case_summary,
            episode_step=self.step_count,
            app_context={
                "task_description": TASK_CONFIG[self.current_task]["description"],
                "policy_required": case["policy_required"],
                "linked_workflow": bool(case["linked_case_ids"]),
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

        actions = [
            ActionTypeEnum.REVIEW_TRANSACTION,
            ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
            ActionTypeEnum.FETCH_MERCHANT_PROFILE,
            ActionTypeEnum.FETCH_NETWORK_GRAPH,
            ActionTypeEnum.CHECK_POLICY,
            ActionTypeEnum.ADD_CASE_NOTE,
            ActionTypeEnum.RESOLVE_CASE,
        ]
        return actions

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
