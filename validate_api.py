#!/usr/bin/env python3
"""Validation script for FraudShield FastAPI endpoints."""

from __future__ import annotations

import sys


def check_fastapi_import():
    print("1. FastAPI Import Check:")
    try:
        from fastapi.testclient import TestClient

        print("   PASS FastAPI imports successfully")
        return True, TestClient
    except Exception as exc:
        print(f"   FAIL FastAPI import failed: {exc}")
        return False, None


def check_server_creation(TestClient):
    print("\n2. Server Creation Check:")
    try:
        from server.app import app

        client = TestClient(app)
        print("   PASS server.app imports successfully")
        print("   PASS TestClient created successfully")
        return True, client
    except Exception as exc:
        print(f"   FAIL Server creation failed: {exc}")
        return False, None


def check_health_endpoint(client):
    print("\n3. Health Endpoint Check:")
    response = client.get("/health")
    print(f"   GET /health -> {response.status_code}")
    if response.status_code != 200:
        return False
    data = response.json()
    print(f"   status={data.get('status')} data_loaded={data.get('data_loaded')}")
    return data.get("status") in {"healthy", "degraded"}


def check_tasks_endpoint(client):
    print("\n4. Tasks Endpoint Check:")
    response = client.get("/tasks")
    print(f"   GET /tasks -> {response.status_code}")
    if response.status_code != 200:
        return False
    data = response.json()
    print(f"   tasks={list(data.keys())}")
    return set(data.keys()) == {"easy", "medium", "hard"}


def check_reset_and_state(client):
    print("\n5. Reset + State Check:")
    response = client.post("/reset?task=hard")
    print(f"   POST /reset?task=hard -> {response.status_code}")
    if response.status_code != 200:
        return False, None
    payload = response.json()
    observation = payload["observation"]
    print(
        "   observation:",
        observation["case_id"],
        observation["task_name"],
        observation["current_screen"],
        observation["allowed_actions"],
    )
    state_response = client.get("/state")
    print(f"   GET /state -> {state_response.status_code}")
    return state_response.status_code == 200, observation["case_id"]


def check_step_flow(client, case_id: str):
    print("\n6. Step Flow Check:")
    review_action = {"case_id": case_id, "action_type": "review_transaction", "reasoning": "Open the case first."}
    review_response = client.post("/step", json=review_action)
    print(f"   review_transaction -> {review_response.status_code}")
    if review_response.status_code != 200:
        return False

    note_action = {
        "case_id": case_id,
        "action_type": "add_case_note",
        "note_text": "Reviewed the case details and documented the initial enterprise workflow findings.",
    }
    note_response = client.post("/step", json=note_action)
    print(f"   add_case_note -> {note_response.status_code}")
    if note_response.status_code != 200:
        return False

    resolve_action = {
        "case_id": case_id,
        "action_type": "resolve_case",
        "resolution": "block",
        "reasoning": "The reviewed case signals justify a blocking decision in this smoke test.",
    }
    resolve_response = client.post("/step", json=resolve_action)
    print(f"   resolve_case -> {resolve_response.status_code}")
    if resolve_response.status_code != 200:
        return False

    reward = resolve_response.json()["reward"]
    print(f"   final reward={reward['value']} correct={reward.get('is_correct')}")
    return True


def check_invalid_payload(client, case_id: str):
    print("\n7. Invalid Payload Check:")
    invalid_action = {"case_id": case_id, "action_type": "resolve_case", "reasoning": "Too short"}
    response = client.post("/step", json=invalid_action)
    print(f"   invalid resolve_case -> {response.status_code}")
    return response.status_code in {400, 422}


def main():
    print("=" * 70)
    print("FraudShield API Validation Suite")
    print("=" * 70)

    success, TestClient = check_fastapi_import()
    if not success:
        return 1

    success, client = check_server_creation(TestClient)
    if not success:
        return 1

    checks = []
    checks.append(("Health endpoint", check_health_endpoint(client)))
    checks.append(("Tasks endpoint", check_tasks_endpoint(client)))
    success, case_id = check_reset_and_state(client)
    checks.append(("Reset + state", success))
    if success and case_id:
        checks.append(("Step flow", check_step_flow(client, case_id)))
        checks.append(("Invalid payload", check_invalid_payload(client, case_id)))

    print("\n" + "=" * 70)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 70)
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
