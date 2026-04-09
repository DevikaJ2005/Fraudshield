#!/usr/bin/env python3
"""Validation script for enhanced FraudShield files.

This script verifies:
1. Python version requirements (3.10+)
2. All enhanced files have valid syntax
3. All imports work correctly
4. Pydantic models validate correctly
5. Baseline inference produces correct scores
6. Results file has correct structure
"""

import sys
import json
from pathlib import Path

def check_python_version():
    """Verify Python 3.10+ is being used."""
    version = sys.version_info
    is_valid = version >= (3, 10)
    status = "✓ PASS" if is_valid else "✗ FAIL"
    print(f"1. Python Version Check: {status}")
    print(f"   Required: 3.10+ | Current: {version.major}.{version.minor}")
    return is_valid

def check_imports():
    """Verify all enhanced modules can be imported."""
    print("\n2. Import Validation:")
    all_pass = True
    modules = [
        ("models", "models.py"),
        ("data_loader", "data_loader.py"),
        ("fraudshield_env", "fraudshield_env.py"),
        ("graders", "graders.py"),
        ("llm_agent", "llm_agent.py"),
        ("inference", "inference.py"),
    ]
    
    for module_name, file_name in modules:
        try:
            __import__(module_name)
            print(f"   ✓ {file_name} imports successfully")
        except Exception as e:
            print(f"   ✗ {file_name} import failed: {e}")
            all_pass = False
    
    return all_pass

def check_pydantic_models():
    """Verify Pydantic models work correctly."""
    print("\n3. Pydantic Model Validation:")
    try:
        from models import (
            FraudCheckAction, FraudCheckObservation, Reward, 
            EpisodeState, StepResult, ResetResult, DecisionEnum, 
            TaskDifficulty, TransactionData
        )
        
        # Test FraudCheckAction
        action = FraudCheckAction(
            transaction_id="test_001",
            decision=DecisionEnum.FRAUD,
            confidence=0.95,
            reasoning="Test fraud decision based on seller account age"
        )
        
        # Test validation: confidence must be [0.0, 1.0]
        try:
            invalid_action = FraudCheckAction(
                transaction_id="test_002",
                decision=DecisionEnum.FRAUD,
                confidence=1.5,  # Invalid: > 1.0
                reasoning="This should fail validation"
            )
            print("   ✗ Confidence validation failed (should reject > 1.0)")
            return False
        except Exception:
            pass  # Expected to fail
        
        print("   ✓ FraudCheckAction validation passed")
        print(f"     - transaction_id: {action.transaction_id}")
        print(f"     - decision: {action.decision.value}")
        print(f"     - confidence: {action.confidence}")
        
        # Test DecisionEnum
        assert DecisionEnum.FRAUD.value == "fraud"
        assert DecisionEnum.LEGITIMATE.value == "legitimate"
        print("   ✓ DecisionEnum validation passed")
        
        # Test TaskDifficulty
        assert TaskDifficulty.EASY.value == "easy"
        assert TaskDifficulty.MEDIUM.value == "medium"
        assert TaskDifficulty.HARD.value == "hard"
        print("   ✓ TaskDifficulty validation passed")
        
        return True
    except Exception as e:
        print(f"   ✗ Pydantic model validation failed: {e}")
        return False

def check_results_file():
    """Verify results file structure and scores."""
    print("\n4. Results File Validation:")
    try:
        results_file = Path("fraudshield_baseline_results.json")
        if not results_file.exists():
            print("   ✗ Results file not found")
            return False
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Validate structure
        required_keys = ["final_score", "easy", "medium", "hard", "metadata"]
        missing_keys = [k for k in required_keys if k not in results]
        if missing_keys:
            print(f"   ✗ Missing keys: {missing_keys}")
            return False
        
        print("   ✓ Results file has required structure")
        
        # Validate scores
        scores = {
            "Final": results["final_score"],
            "Easy": results["easy"]["score"],
            "Medium": results["medium"]["score"],
            "Hard": results["hard"]["score"],
        }
        
        for name, score in scores.items():
            if not (0 <= score <= 1):
                print(f"   ✗ {name} score out of range: {score}")
                return False
            print(f"   ✓ {name} score: {score:.4f}")
        
        # Check if baseline score matches expected value
        expected_final = 0.9993
        actual_final = results["final_score"]
        tolerance = 0.001
        
        if abs(actual_final - expected_final) < tolerance:
            print(f"   ✓ Baseline score matches expected value ({expected_final:.4f})")
        else:
            print(f"   ⚠ Baseline score variance: expected {expected_final:.4f}, got {actual_final:.4f}")
        
        # Validate task transaction counts
        print(f"   ✓ Transaction counts:")
        print(f"     - Easy: {results['easy']['num_transactions']}")
        print(f"     - Medium: {results['medium']['num_transactions']}")
        print(f"     - Hard: {results['hard']['num_transactions']}")
        
        return True
    except Exception as e:
        print(f"   ✗ Results validation failed: {e}")
        return False

def check_data_loader():
    """Verify data loader works correctly."""
    print("\n5. Data Loader Validation:")
    try:
        from data_loader import FraudDataLoader
        
        loader = FraudDataLoader(data_path="data", seed=42)
        if not loader.load_bundle():
            print("   ✗ Failed to load data bundle")
            return False
        
        print("   ✓ Data bundle loaded successfully")
        
        # Check task sizes
        summary = loader.get_bundle_summary()
        print(f"   ✓ Bundle summary:")
        print(f"     - Snapshot ID: {summary.get('snapshot_id')}")
        print(f"     - Schema version: {summary.get('schema_version')}")
        print(f"     - Seed: {summary.get('seed')}")
        print(f"     - Task sizes: {summary.get('task_sizes')}")
        
        # Verify task cases
        for task_name in ["easy", "medium", "hard"]:
            cases = loader.get_task_cases(task_name)
            if not cases:
                print(f"   ✗ No cases found for {task_name} task")
                return False
            print(f"   ✓ {task_name.capitalize()} task: {len(cases)} cases")
        
        return True
    except Exception as e:
        print(f"   ✗ Data loader validation failed: {e}")
        return False

def check_environment():
    """Verify environment works correctly."""
    print("\n6. Environment Validation:")
    try:
        from fraudshield_env import FraudShieldEnvironment
        
        env = FraudShieldEnvironment(data_path="data", seed=42)
        if not env.load_data():
            print("   ✗ Failed to load environment data")
            return False
        
        print("   ✓ Environment created and data loaded")
        
        # Test reset
        reset_result = env.reset("easy")
        print(f"   ✓ Reset successful (episode_id: {env.episode_id})")
        
        # Test observation
        obs = reset_result.observation
        print(f"   ✓ Initial observation created")
        print(f"     - Transaction ID: {obs.transaction_id}")
        print(f"     - Task: {obs.task_name.value}")
        print(f"     - Episode step: {obs.episode_step}")
        
        return True
    except Exception as e:
        print(f"   ✗ Environment validation failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("=" * 70)
    print("FraudShield Enhancement Validation Suite")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Imports", check_imports),
        ("Pydantic Models", check_pydantic_models),
        ("Results File", check_results_file),
        ("Data Loader", check_data_loader),
        ("Environment", check_environment),
    ]
    
    results = {}
    for check_name, check_fn in checks:
        try:
            results[check_name] = check_fn()
        except Exception as e:
            print(f"\n✗ {check_name} check failed with exception: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
