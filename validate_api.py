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


def check_info_and_tasks(client):
    print("\n4. Info + Tasks Check:")
    info = client.get("/info")
    print(f"   GET /info -> {info.status_code}")
    if info.status_code != 200:
        return False

    tasks = client.get("/tasks")
    print(f"   GET /tasks -> {tasks.status_code}")
    if tasks.status_code != 200:
        return False
    task_payload = tasks.json()
    print(f"   tasks={list(task_payload.keys())}")
    return set(task_payload.keys()) == {"easy", "medium", "hard"}


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

    network_action = {
        "case_id": case_id,
        "action_type": "fetch_network_graph",
        "reasoning": "Reveal linked activity before final routing.",
    }
    network_response = client.post("/step", json=network_action)
    print(f"   fetch_network_graph -> {network_response.status_code}")
    if network_response.status_code != 200:
        return False

    note_action = {
        "case_id": case_id,
        "action_type": "add_case_note",
        "note_text": "Reviewed the transaction trace and hidden evidence before selecting the final route.",
    }
    note_response = client.post("/step", json=note_action)
    print(f"   add_case_note -> {note_response.status_code}")
    if note_response.status_code != 200:
        return False

    resolve_action = {
        "case_id": case_id,
        "action_type": "resolve_case",
        "resolution": "block",
        "reasoning": "The reviewed case signals justify a blocking decision in this API smoke test.",
    }
    resolve_response = client.post("/step", json=resolve_action)
    print(f"   resolve_case -> {resolve_response.status_code}")
    if resolve_response.status_code != 200:
        return False

    reward = resolve_response.json()["reward"]
    print(f"   final reward={reward['value']} correct={reward.get('is_correct')}")
    return True


def check_schema_and_metadata(client):
    print("\n7. Schema + Metadata Check:")
    metadata = client.get("/metadata")
    schema = client.get("/schema")
    print(f"   GET /metadata -> {metadata.status_code}")
    print(f"   GET /schema -> {schema.status_code}")
    if metadata.status_code != 200 or schema.status_code != 200:
        return False
    metadata_payload = metadata.json()
    print(f"   workflow_views={metadata_payload.get('workflow_views')}")
    return metadata_payload.get("name") == "fraudshield"


def check_invalid_payload(client, case_id: str):
    print("\n8. Invalid Payload Check:")
    invalid_action = {"case_id": case_id, "action_type": "resolve_case", "reasoning": "Too short"}
    response = client.post("/step", json=invalid_action)
    print(f"   invalid resolve_case -> {response.status_code}")
    return response.status_code in {400, 422}


def check_mcp(client):
    print("\n9. MCP Check:")
    initialize = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    print(f"   initialize -> {initialize.status_code}")
    if initialize.status_code != 200:
        return False

    tool_list = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    print(f"   tools/list -> {tool_list.status_code}")
    if tool_list.status_code != 200:
        return False
    tools = tool_list.json()["result"]["tools"]
    print(f"   tool_count={len(tools)}")

    tool_call = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "environment.tasks", "arguments": {}},
        },
    )
    print(f"   tools/call environment.tasks -> {tool_call.status_code}")
    return tool_call.status_code == 200


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
    checks.append(("Info + tasks", check_info_and_tasks(client)))
    success, case_id = check_reset_and_state(client)
    checks.append(("Reset + state", success))
    checks.append(("Schema + metadata", check_schema_and_metadata(client)))
    if success and case_id:
        checks.append(("Step flow", check_step_flow(client, case_id)))
        checks.append(("Invalid payload", check_invalid_payload(client, case_id)))
    checks.append(("MCP flow", check_mcp(client)))

    print("\n" + "=" * 70)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 70)
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
