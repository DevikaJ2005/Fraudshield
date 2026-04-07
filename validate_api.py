#!/usr/bin/env python3
"""Validation script for FastAPI server endpoints.

This script tests:
1. FastAPI app can be created
2. All endpoints can be defined
3. Health check endpoint works
4. Lifespan events work
5. Exception handling works
6. Pydantic request/response models work with FastAPI
"""

import sys
import json
from pathlib import Path

def check_fastapi_import():
    """Verify FastAPI can be imported."""
    print("1. FastAPI Import Check:")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        print("   ✓ FastAPI imports successfully")
        return True, TestClient
    except Exception as e:
        print(f"   ✗ FastAPI import failed: {e}")
        return False, None

def check_server_creation(TestClient):
    """Verify server/app.py can be imported and app created."""
    print("\n2. Server Creation Check:")
    try:
        from server.app import app
        print("   ✓ server/app.py imports successfully")
        
        client = TestClient(app)
        print("   ✓ TestClient created successfully")
        return True, app, client
    except Exception as e:
        print(f"   ✗ Server creation failed: {e}")
        return False, None, None

def check_health_endpoint(client):
    """Test the /health endpoint."""
    print("\n3. Health Endpoint Check:")
    try:
        response = client.get("/health")
        print(f"   ✓ GET /health: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"     - Status: {data.get('status')}")
            print(f"     - Service: {data.get('service')}")
            print(f"     - Data loaded: {data.get('data_loaded')}")
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Health endpoint test failed: {e}")
        return False

def check_root_endpoint(client):
    """Test the root endpoint."""
    print("\n4. Root Endpoint Check:")
    try:
        response = client.get("/")
        print(f"   ✓ GET /: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"     - Service: {data.get('service')}")
            print(f"     - Version: {data.get('version')}")
            endpoints = data.get('endpoints', {})
            print(f"     - Endpoints defined: {len(endpoints)}")
            for endpoint_name, endpoint_path in endpoints.items():
                print(f"       • {endpoint_name}: {endpoint_path}")
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Root endpoint test failed: {e}")
        return False

def check_info_endpoint(client):
    """Test the /info endpoint."""
    print("\n5. Info Endpoint Check:")
    try:
        response = client.get("/info")
        print(f"   ✓ GET /info: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"     - Name: {data.get('name')}")
            print(f"     - Version: {data.get('version')}")
            tasks = data.get('tasks', {})
            print(f"     - Tasks defined: {len(tasks)}")
            for task_name, max_steps in tasks.items():
                print(f"       • {task_name}: {max_steps} steps")
            
            snapshot = data.get('data_snapshot', {})
            if snapshot:
                print(f"     - Data snapshot:")
                print(f"       • ID: {snapshot.get('snapshot_id')}")
                print(f"       • Schema version: {snapshot.get('schema_version')}")
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Info endpoint test failed: {e}")
        return False

def check_tasks_endpoint(client):
    """Test the /tasks endpoint."""
    print("\n6. Tasks Endpoint Check:")
    try:
        response = client.get("/tasks")
        print(f"   ✓ GET /tasks: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"     - Tasks defined: {len(data)}")
            for task_name, task_info in data.items():
                print(f"       • {task_name}:")
                print(f"         - Difficulty: {task_info.get('difficulty')}")
                print(f"         - Transactions: {task_info.get('num_transactions')}")
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Tasks endpoint test failed: {e}")
        return False

def check_reset_endpoint(client):
    """Test the /reset endpoint."""
    print("\n7. Reset Endpoint Check:")
    try:
        # Test with default (easy)
        response = client.post("/reset")
        print(f"   ✓ POST /reset: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_keys = ['observation', 'info', 'episode_id']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"   ✗ Missing keys: {missing_keys}")
                return False
            
            print(f"     - Episode ID: {data.get('episode_id')}")
            print(f"     - Observation fields: {len(data.get('observation', {}))}")
            
            obs = data.get('observation', {})
            print(f"       • Transaction ID: {obs.get('transaction_id')}")
            print(f"       • Task: {obs.get('task_name')}")
            
            info = data.get('info', {})
            print(f"     - Episode info:")
            print(f"       • Task: {info.get('task')}")
            print(f"       • Max steps: {info.get('max_steps')}")
            
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Reset endpoint test failed: {e}")
        return False

def check_step_endpoint(client):
    """Test the /step endpoint."""
    print("\n8. Step Endpoint Check:")
    try:
        # First reset to get into valid state
        reset_response = client.post("/reset?task=easy")
        if reset_response.status_code != 200:
            print("   ✗ Reset failed, cannot test step")
            return False
        
        # Now test step with a valid action
        action = {
            "transaction_id": reset_response.json()['observation']['transaction_id'],
            "decision": "fraud",
            "confidence": 0.85,
            "reasoning": "Seller account age is suspicious and amount is high"
        }
        
        response = client.post("/step", json=action)
        print(f"   ✓ POST /step: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_keys = ['observation', 'reward', 'done', 'info']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"   ✗ Missing keys: {missing_keys}")
                return False
            
            reward = data.get('reward', {})
            print(f"     - Reward: {reward.get('value'):.2f}")
            print(f"     - Is correct: {reward.get('is_correct')}")
            print(f"     - Ground truth: {reward.get('ground_truth')}")
            print(f"     - Done: {data.get('done')}")
            
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Step endpoint test failed: {e}")
        return False

def check_state_endpoint(client):
    """Test the /state endpoint."""
    print("\n9. State Endpoint Check:")
    try:
        # First reset to get into valid state
        reset_response = client.post("/reset?task=medium")
        if reset_response.status_code != 200:
            print("   ✗ Reset failed, cannot test state")
            return False
        
        response = client.get("/state")
        print(f"   ✓ GET /state: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_keys = ['episode_id', 'task_name', 'step_count', 'is_done']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"   ✗ Missing keys: {missing_keys}")
                return False
            
            print(f"     - Episode ID: {data.get('episode_id')}")
            print(f"     - Task: {data.get('task_name')}")
            print(f"     - Step count: {data.get('step_count')}")
            print(f"     - Done: {data.get('is_done')}")
            
            return True
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ State endpoint test failed: {e}")
        return False

def check_invalid_input(client):
    """Test error handling for invalid input."""
    print("\n10. Error Handling Check:")
    try:
        # Test invalid action (confidence out of range)
        reset_response = client.post("/reset?task=easy")
        obs_id = reset_response.json()['observation']['transaction_id']
        
        invalid_action = {
            "transaction_id": obs_id,
            "decision": "fraud",
            "confidence": 1.5,  # Invalid: > 1.0
            "reasoning": "This should fail"
        }
        
        response = client.post("/step", json=invalid_action)
        
        # Should return 422 (validation error) not 200
        if response.status_code in [400, 422]:
            print(f"   ✓ Invalid input correctly rejected: {response.status_code}")
            error_data = response.json()
            print(f"     - Error caught correctly")
            return True
        else:
            print(f"   ✗ Expected validation error but got {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False

def main():
    """Run all API validation checks."""
    print("=" * 70)
    print("FraudShield API Server Validation Suite")
    print("=" * 70)
    
    success, TestClient = check_fastapi_import()
    if not success:
        print("\n✗ Cannot continue without FastAPI")
        return 1
    
    success, app, client = check_server_creation(TestClient)
    if not success:
        print("\n✗ Cannot continue without working server")
        return 1
    
    checks = [
        ("Health endpoint", lambda: check_health_endpoint(client)),
        ("Root endpoint", lambda: check_root_endpoint(client)),
        ("Info endpoint", lambda: check_info_endpoint(client)),
        ("Tasks endpoint", lambda: check_tasks_endpoint(client)),
        ("Reset endpoint", lambda: check_reset_endpoint(client)),
        ("Step endpoint", lambda: check_step_endpoint(client)),
        ("State endpoint", lambda: check_state_endpoint(client)),
        ("Error handling", lambda: check_invalid_input(client)),
    ]
    
    results = {}
    for check_name, check_fn in checks:
        try:
            results[check_name] = check_fn()
        except Exception as e:
            print(f"\n✗ {check_name} failed with exception: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 70)
    print("API VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    if all_passed:
        print("✓ ALL API VALIDATIONS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME API VALIDATIONS FAILED")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
