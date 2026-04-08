"""FraudShield environment implementation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List

from data_loader import FraudDataLoader
from models import (
    ActionTypeEnum,
    DecisionEnum,
    EpisodeState,
    FraudCheckAction,
    FraudCheckObservation,
    InvestigationTargetEnum,
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
        self.decisions_made = 0
        self.investigations_used = 0
        self.current_case_budget_remaining = 0
        self.current_revealed_evidence: Dict[str, Dict[str, Any]] = {}
        self.current_investigation_catalog: Dict[str, Dict[str, Any]] = {}
        self.current_case_stage = "triage"
        self.current_visible_signal_summary = ""

        self.case_counts = {
            TaskDifficulty.EASY: 24,
            TaskDifficulty.MEDIUM: 36,
            TaskDifficulty.HARD: 48,
        }
        self.investigation_budget_per_case = {
            TaskDifficulty.EASY: 1,
            TaskDifficulty.MEDIUM: 2,
            TaskDifficulty.HARD: 3,
        }
        self.max_steps = {
            task: self.case_counts[task] * (1 + self.investigation_budget_per_case[task])
            for task in TaskDifficulty
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
        self.decisions_made = 0
        self.investigations_used = 0
        self.current_case_budget_remaining = 0
        self.current_revealed_evidence = {}
        self.current_investigation_catalog = {}
        self.current_case_stage = "triage"
        self.current_visible_signal_summary = ""

        self.current_cases = self.data_loader.get_task_cases(task)
        self.ground_truth_labels = [case["label"] for case in self.current_cases]
        self.case_counts[self.current_task] = len(self.current_cases)
        self.max_steps[self.current_task] = len(self.current_cases) * (
            1 + self.investigation_budget_per_case[self.current_task]
        )
        self._prepare_current_case_state()

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

        if action.action_type == ActionTypeEnum.INVESTIGATE:
            return self._handle_investigation_action(
                action=action,
                current_case=current_case,
                expected_transaction_id=expected_transaction_id,
                wrong_transaction_id=wrong_transaction_id,
                risk_score=risk_score,
                business_cost=business_cost,
            )

        if action.decision is None or action.confidence is None:
            raise ValueError("Decision actions must include both decision and confidence.")

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
        self.decisions_made += 1
        self.current_transaction_idx += 1
        self.is_done = self.current_transaction_idx >= len(self.current_cases)

        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            is_correct=is_correct,
            ground_truth=DecisionEnum(ground_truth),
            confidence_penalty=confidence_penalty,
            business_impact=business_cost,
            action_type=ActionTypeEnum.DECIDE,
            action_cost=0.0,
        )

        if self.is_done:
            observation = self._get_terminal_observation()
        else:
            self._prepare_current_case_state()
            observation = self._get_observation()

        info = {
            "step": self.step_count,
            "accuracy_so_far": round(self.correct_predictions / max(1, self.decisions_made), 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "expected_transaction_id": expected_transaction_id,
            "wrong_transaction_id": wrong_transaction_id,
            "risk_score": risk_score,
            "business_cost": business_cost,
            "action_type": action.action_type.value,
            "investigation_budget_remaining": self.current_case_budget_remaining,
        }
        return StepResult(observation=observation, reward=reward, done=self.is_done, info=info)

    def state(self) -> EpisodeState:
        """Return the current episode state."""

        return EpisodeState(
            episode_id=self.episode_id,
            task_name=self.current_task,
            step_count=self.step_count,
            transactions_evaluated=self.decisions_made,
            cumulative_reward=self.cumulative_reward,
            correct_predictions=self.correct_predictions,
            is_done=self.is_done,
            max_steps=self.max_steps[self.current_task],
            current_transaction_id=self._current_transaction_id(),
            investigations_used=self.investigations_used,
            investigation_budget_remaining=self.current_case_budget_remaining,
            revealed_evidence_keys=sorted(self.current_revealed_evidence.keys()),
        )

    def _handle_investigation_action(
        self,
        action: FraudCheckAction,
        current_case: Dict[str, Any],
        expected_transaction_id: str,
        wrong_transaction_id: bool,
        risk_score: float,
        business_cost: float,
    ) -> StepResult:
        """Reveal optional evidence for the current case without finalizing a verdict."""

        target = action.investigation_target
        if target is None:
            raise ValueError("Investigation actions must specify an investigation_target.")

        reward_value, action_cost, reward_reason, evidence_revealed = self._calculate_investigation_reward(
            target=target,
            current_case=current_case,
            risk_score=risk_score,
            wrong_transaction_id=wrong_transaction_id,
        )

        if (
            not wrong_transaction_id
            and evidence_revealed
            and target.value not in self.current_revealed_evidence
            and target.value in self.current_investigation_catalog
        ):
            self.current_revealed_evidence[target.value] = self.current_investigation_catalog[target.value]
            self.current_case_budget_remaining -= 1
            self.investigations_used += 1

        self.current_case_stage = self._derive_case_stage()
        self.cumulative_reward += reward_value
        self.step_count += 1

        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            is_correct=None,
            ground_truth=None,
            confidence_penalty=0.0,
            business_impact=business_cost,
            action_type=ActionTypeEnum.INVESTIGATE,
            investigation_target=target,
            evidence_revealed=evidence_revealed,
            action_cost=action_cost,
        )
        observation = self._get_observation()
        info = {
            "step": self.step_count,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "expected_transaction_id": expected_transaction_id,
            "wrong_transaction_id": wrong_transaction_id,
            "risk_score": risk_score,
            "business_cost": business_cost,
            "action_type": action.action_type.value,
            "investigation_target": target.value,
            "investigation_budget_remaining": self.current_case_budget_remaining,
            "revealed_evidence_keys": sorted(self.current_revealed_evidence.keys()),
            "case_stage": self.current_case_stage,
        }
        return StepResult(observation=observation, reward=reward, done=False, info=info)

    def _prepare_current_case_state(self) -> None:
        """Initialize deterministic investigation context for the active case."""

        if self.current_transaction_idx >= len(self.current_cases):
            self.current_case_budget_remaining = 0
            self.current_revealed_evidence = {}
            self.current_investigation_catalog = {}
            self.current_visible_signal_summary = ""
            self.current_case_stage = "completed"
            return

        current_case = self.current_cases[self.current_transaction_idx]
        self.current_case_budget_remaining = self.investigation_budget_per_case[self.current_task]
        self.current_revealed_evidence = {}
        self.current_investigation_catalog = self._build_investigation_catalog(current_case)
        self.current_visible_signal_summary = self._build_visible_signal_summary(current_case)
        self.current_case_stage = self._derive_case_stage()

    def _derive_case_stage(self) -> str:
        """Describe where the current review sits in the investigation workflow."""

        if self.is_done:
            return "completed"
        if self.current_revealed_evidence and self.current_case_budget_remaining == 0:
            return "awaiting_decision"
        if self.current_revealed_evidence:
            return "investigating"
        if self.current_case_budget_remaining == 0:
            return "decision_only"
        return "triage"

    def _current_transaction_id(self) -> str:
        """Return the active transaction identifier or TERMINAL if the episode is done."""

        if self.is_done or self.current_transaction_idx >= len(self.current_cases):
            return "TERMINAL"
        return str(self.current_cases[self.current_transaction_idx]["transaction_id"])

    def _calculate_investigation_reward(
        self,
        target: InvestigationTargetEnum,
        current_case: Dict[str, Any],
        risk_score: float,
        wrong_transaction_id: bool,
    ) -> tuple[float, float, str, bool]:
        """Apply a small cost-sensitive reward for requesting more evidence."""

        target_key = target.value
        if wrong_transaction_id:
            return -0.12, 0.08, "investigation referenced the wrong transaction_id", False
        if self.current_case_budget_remaining <= 0:
            return -0.08, 0.06, "investigation budget is exhausted for this case", False
        if target_key in self.current_revealed_evidence:
            return -0.04, 0.02, f"{target_key} was already revealed earlier in the review", False
        if target_key not in self.current_investigation_catalog:
            return -0.06, 0.04, f"{target_key} is not a valid investigation target", False

        ambiguity = 1.0 - abs((risk_score * 2.0) - 1.0)
        base_cost = {
            TaskDifficulty.EASY: 0.05,
            TaskDifficulty.MEDIUM: 0.04,
            TaskDifficulty.HARD: 0.03,
        }[self.current_task]
        recommended_targets = self._recommended_investigation_targets(current_case)
        recommendation_bonus = 0.025 if target_key in recommended_targets else 0.0
        ambiguity_bonus = {
            TaskDifficulty.EASY: 0.01,
            TaskDifficulty.MEDIUM: 0.03,
            TaskDifficulty.HARD: 0.05,
        }[self.current_task] * ambiguity
        reward_value = max(-0.12, min(0.12, recommendation_bonus + ambiguity_bonus - base_cost))
        reward_reason = (
            f"investigated={target_key}, ambiguity={ambiguity:.2f}, "
            f"budget_remaining_after={self.current_case_budget_remaining - 1}"
        )
        return reward_value, base_cost, reward_reason, True

    def _recommended_investigation_targets(self, current_case: Dict[str, Any]) -> List[str]:
        """Return evidence bundles that are especially relevant for the current transaction."""

        txn = current_case["transaction_data"]
        history = current_case["historical_context"]
        targets: List[str] = []

        if txn["device_country"] != txn["shipping_address"] or txn["shared_device_accounts_24h"] >= 4:
            targets.append(InvestigationTargetEnum.DEVICE_INTEL.value)
        if txn["seller_chargeback_rate_30d"] >= 0.08 or txn["buyer_disputes_90d"] >= 2:
            targets.append(InvestigationTargetEnum.PAYMENT_TRACE.value)
        if history.get("cluster_alert_score", 0.0) >= 0.65 or txn["previous_fraud_flags"] > 0:
            targets.append(InvestigationTargetEnum.NETWORK_GRAPH.value)
        if txn["shipping_speed"] == "overnight" or txn["same_address_orders_24h"] >= 4:
            targets.append(InvestigationTargetEnum.FULFILLMENT_REVIEW.value)
        if txn["seller_avg_rating"] < 4.2 or history.get("recent_refunds_7d", 0) >= 3:
            targets.append(InvestigationTargetEnum.TRUST_NOTES.value)

        if not targets:
            targets.append(InvestigationTargetEnum.TRUST_NOTES.value)
        return targets

    def _build_visible_signal_summary(self, current_case: Dict[str, Any]) -> str:
        """Create a concise triage summary from the visible structured features."""

        txn = current_case["transaction_data"]
        history = current_case["historical_context"]
        highlights: List[str] = []

        if txn["device_country"] != txn["shipping_address"]:
            highlights.append("device/shipping geography mismatch")
        if txn["seller_account_age_days"] <= 30:
            highlights.append("new seller account")
        if txn["seller_chargeback_rate_30d"] >= 0.08:
            highlights.append("elevated seller chargeback rate")
        if txn["shared_device_accounts_24h"] >= 4:
            highlights.append("device reused across multiple accounts")
        if txn["same_address_orders_24h"] >= 4:
            highlights.append("address velocity spike")
        if history.get("cluster_alert_score", 0.0) >= 0.65:
            highlights.append("network alert score elevated")
        if txn["is_repeat_buyer"] and txn["seller_avg_rating"] >= 4.5:
            highlights.append("repeat buyer with strong seller trust history")

        if not highlights:
            highlights.append("signals are mixed with no single decisive fraud marker")

        top_highlights = ", ".join(highlights[:3])
        return (
            f"Triage summary: {top_highlights}. "
            f"Visible amount=${txn['amount']:.2f}, category={txn['item_category']}, "
            f"shipping={txn['shipping_speed']}."
        )

    def _build_investigation_catalog(self, current_case: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate deterministic evidence bundles for the active case."""

        txn = current_case["transaction_data"]
        history = current_case["historical_context"]
        device_mismatch = txn["device_country"] != txn["shipping_address"]
        amount_gap = txn["amount"] / max(txn["item_price"], 1.0)
        device_risk = float(min(0.99, current_case["risk_score"] + (0.12 if device_mismatch else -0.05)))
        payment_risk = float(min(0.99, current_case["risk_score"] + txn["seller_chargeback_rate_30d"] * 0.5))
        network_risk = float(min(0.99, history.get("cluster_alert_score", 0.0) + txn["previous_fraud_flags"] * 0.06))

        return {
            InvestigationTargetEnum.DEVICE_INTEL.value: {
                "headline": "Device and session intelligence",
                "risk_band": self._risk_band(device_risk),
                "details": {
                    "device_country_match": not device_mismatch,
                    "device_reuse_accounts_24h": txn["shared_device_accounts_24h"],
                    "proxy_suspected": device_mismatch or txn["shared_device_accounts_24h"] >= 5,
                    "device_match_history": history.get("device_match", True),
                },
                "narrative": (
                    "Device telemetry shows "
                    f"{'a country mismatch' if device_mismatch else 'a country match'}, "
                    f"with reuse across {txn['shared_device_accounts_24h']} account(s) in the last day."
                ),
            },
            InvestigationTargetEnum.PAYMENT_TRACE.value: {
                "headline": "Payment and issuer trace",
                "risk_band": self._risk_band(payment_risk),
                "details": {
                    "avs_status": "mismatch" if device_mismatch or amount_gap >= 1.18 else "match",
                    "cvv_result": "review" if txn["previous_fraud_flags"] > 0 else "match",
                    "issuer_velocity_alert": txn["buyer_disputes_90d"] + txn["previous_fraud_flags"],
                    "prepaid_card_risk": txn["payment_method"] in {"wallet", "prepaid", "gift_card"},
                },
                "narrative": (
                    f"Payment review found issuer alert count {txn['buyer_disputes_90d'] + txn['previous_fraud_flags']} "
                    f"and a price-to-amount gap of {amount_gap:.2f}x."
                ),
            },
            InvestigationTargetEnum.NETWORK_GRAPH.value: {
                "headline": "Linked-entity and ring analysis",
                "risk_band": self._risk_band(network_risk),
                "details": {
                    "cluster_alert_score": history.get("cluster_alert_score", 0.0),
                    "linked_cards_7d": history.get("linked_cards_7d", 0),
                    "recent_related_flags": txn["previous_fraud_flags"],
                    "seller_transactions_1h": history.get("seller_transactions_1h", 0),
                },
                "narrative": (
                    "The network graph shows "
                    f"{history.get('linked_cards_7d', 0)} linked card(s) in 7 days and "
                    f"cluster score {history.get('cluster_alert_score', 0.0):.2f}."
                ),
            },
            InvestigationTargetEnum.FULFILLMENT_REVIEW.value: {
                "headline": "Fulfillment and address review",
                "risk_band": self._risk_band(float(min(0.99, current_case["risk_score"] + txn["same_address_orders_24h"] * 0.03))),
                "details": {
                    "shipping_speed": txn["shipping_speed"],
                    "same_address_orders_24h": txn["same_address_orders_24h"],
                    "address_velocity_band": self._risk_band(min(0.99, txn["same_address_orders_24h"] / 8.0)),
                    "geo_mismatch": device_mismatch,
                },
                "narrative": (
                    f"Fulfillment review shows {txn['same_address_orders_24h']} order(s) to the same address "
                    f"in 24h with {txn['shipping_speed']} shipping."
                ),
            },
            InvestigationTargetEnum.TRUST_NOTES.value: {
                "headline": "Trust and safety analyst notes",
                "risk_band": self._risk_band(float(min(0.99, current_case["risk_score"] + history.get("recent_refunds_7d", 0) * 0.04))),
                "details": {
                    "seller_note": self._seller_note(txn, history),
                    "buyer_note": self._buyer_note(txn),
                    "ops_note": self._ops_note(txn, history),
                },
                "narrative": (
                    "Manual notes summarize merchant reputation, recent customer friction, "
                    "and whether the case resembles known abuse patterns."
                ),
            },
        }

    @staticmethod
    def _risk_band(score: float) -> str:
        """Convert a normalized score to a readable risk band."""

        if score >= 0.8:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    @staticmethod
    def _seller_note(txn: Dict[str, Any], history: Dict[str, Any]) -> str:
        """Create a short seller-facing trust note."""

        if txn["seller_account_age_days"] <= 14:
            return "Seller storefront is very new and has limited transaction history."
        if txn["seller_avg_rating"] >= 4.6 and txn["num_seller_reviews"] >= 500:
            return "Seller has strong reputation history with mature review coverage."
        if history.get("recent_refunds_7d", 0) >= 4:
            return "Seller saw an unusual increase in refunds over the last week."
        return "Seller profile is mixed: stable history but not enough evidence for a clean pass."

    @staticmethod
    def _buyer_note(txn: Dict[str, Any]) -> str:
        """Create a short buyer-facing trust note."""

        if txn["buyer_disputes_90d"] >= 3:
            return "Buyer has a recent dispute pattern that resembles friendly-fraud escalation."
        if txn["is_repeat_buyer"]:
            return "Buyer has completed prior purchases with this seller."
        return "Buyer has limited longitudinal history with this merchant."

    @staticmethod
    def _ops_note(txn: Dict[str, Any], history: Dict[str, Any]) -> str:
        """Create a short operations note summarizing ecosystem signals."""

        if history.get("cluster_alert_score", 0.0) >= 0.75:
            return "Ops graph flags this entity cluster as overlapping with a known abuse pattern."
        if txn["same_address_orders_24h"] >= 5:
            return "Operations notes indicate same-address order velocity that merits a closer look."
        return "Operations did not auto-escalate the case, but analysts flagged it as non-routine."

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
        available_targets = [
            InvestigationTargetEnum(target_key)
            for target_key in self.current_investigation_catalog
            if target_key not in self.current_revealed_evidence
        ]
        return FraudCheckObservation(
            transaction_id=current_case["transaction_id"],
            transaction_data=TransactionData(**current_case["transaction_data"]),
            task_name=self.current_task,
            episode_step=self.step_count + 1,
            historical_context=current_case["historical_context"],
            visible_signal_summary=self.current_visible_signal_summary,
            available_investigations=available_targets,
            revealed_evidence=self.current_revealed_evidence,
            investigation_budget_remaining=self.current_case_budget_remaining,
            case_stage=self.current_case_stage,
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
            visible_signal_summary="Episode complete. No active transaction remains for review.",
            available_investigations=[],
            revealed_evidence={},
            investigation_budget_remaining=0,
            case_stage="completed",
        )
