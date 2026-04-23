#!/usr/bin/env python3
"""Validation script for the FraudShield FraudOps environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_python_version():
    print("1. Python Version Check:")
    print(f"   PASS Python {sys.version.split()[0]}")
    return True


def check_imports():
    print("\n2. Import Check:")
    try:
        import fraudshield_env  # noqa: F401
        import graders  # noqa: F401
        import llm_agent  # noqa: F401
        import models  # noqa: F401
        from server import app  # noqa: F401

        print("   PASS Core modules import successfully")
        return True
    except Exception as exc:
        print(f"   FAIL Import error: {exc}")
        return False


def check_models():
    print("\n3. Model Validation Check:")
    try:
        from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum

        note_action = FraudCheckAction(
            case_id="medium_case_01",
            action_type=ActionTypeEnum.ADD_CASE_NOTE,
            note_text="Documented the profile review and policy findings before routing this case.",
        )
        resolve_action = FraudCheckAction(
            case_id="hard_case_primary",
            action_type=ActionTypeEnum.RESOLVE_CASE,
            resolution=ResolutionEnum.ESCALATE,
            reasoning="Linked fraud evidence and policy thresholds justify escalation.",
        )
        print(f"   PASS Note action validated: {note_action.action_type.value}")
        print(f"   PASS Resolve action validated: {resolve_action.resolution.value}")
        return True
    except Exception as exc:
        print(f"   FAIL Model validation error: {exc}")
        return False


def check_data_bundle():
    print("\n4. Snapshot Bundle Check:")
    try:
        path = Path("data/fraudshield_cases.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"   PASS snapshot_id={payload['metadata']['snapshot_id']}")
        print(f"   PASS tasks={list(payload['tasks'].keys())}")
        return True
    except Exception as exc:
        print(f"   FAIL Snapshot read error: {exc}")
        return False


def check_environment():
    print("\n5. Environment Workflow Check:")
    try:
        from fraudshield_env import FraudShieldEnvironment
        from models import ActionTypeEnum, FraudCheckAction, ResolutionEnum

        env = FraudShieldEnvironment(data_path="data", seed=42)
        if not env.load_data():
            print("   FAIL Environment data load failed")
            return False

        reset_result = env.reset("medium")
        case_id = reset_result.observation.case_id
        print(f"   PASS reset -> case_id={case_id} screen={reset_result.observation.current_screen.value}")

        review = env.step(
            FraudCheckAction(case_id=case_id, action_type=ActionTypeEnum.REVIEW_TRANSACTION, reasoning="Open case")
        )
        print(f"   PASS review -> reward={review.reward.value}")

        profile = env.step(
            FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
                reasoning="Need customer context",
            )
        )
        print(f"   PASS customer profile -> reward={profile.reward.value}")

        policy = env.step(
            FraudCheckAction(case_id=case_id, action_type=ActionTypeEnum.CHECK_POLICY, reasoning="Check rules")
        )
        print(f"   PASS policy -> reward={policy.reward.value}")

        note = env.step(
            FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.ADD_CASE_NOTE,
                note_text="Captured customer context and policy triggers before routing this medium case.",
            )
        )
        print(f"   PASS note -> reward={note.reward.value}")

        resolve = env.step(
            FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.RESOLVE_CASE,
                resolution=ResolutionEnum.REQUEST_DOCS,
                reasoning="The profile and policy evidence support a request for documents.",
            )
        )
        print(f"   PASS resolve -> reward={resolve.reward.value} done={resolve.done}")

        report = env.get_episode_report()
        print(f"   PASS report metrics={report['metrics']}")
        return True
    except Exception as exc:
        print(f"   FAIL Environment workflow error: {exc}")
        return False


def main():
    print("=" * 70)
    print("FraudShield Enhancement Validation Suite")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version()),
        ("Imports", check_imports()),
        ("Models", check_models()),
        ("Snapshot bundle", check_data_bundle()),
        ("Environment workflow", check_environment()),
    ]

    print("\n" + "=" * 70)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 70)
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
