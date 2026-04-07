#!/usr/bin/env python3
"""
HuggingFace Spaces Deployment Checklist for FraudShield

This script verifies all required files are present and ready for deployment.
Run this before deploying to confirm everything is set up correctly.
"""

import os
import json
from pathlib import Path

def check_deployment_readiness():
    """Verify all files required for HF Space deployment."""
    
    print("=" * 80)
    print("FraudShield HF Spaces Deployment Readiness Check")
    print("=" * 80)
    print()
    
    checks = {
        "Docker Files": [
            ("Dockerfile", "Core container definition"),
            (".dockerignore", "Build context optimization"),
        ],
        "Python Files": [
            ("server/app.py", "FastAPI application"),
            ("fraudshield_env.py", "Environment core"),
            ("models.py", "Pydantic models"),
            ("inference.py", "Baseline inference"),
            ("data_loader.py", "Data pipeline"),
            ("graders.py", "Scoring functions"),
            ("llm_agent.py", "Agent implementations"),
        ],
        "Configuration Files": [
            ("pyproject.toml", "Project metadata"),
            ("openenv.yaml", "OpenEnv specification"),
        ],
        "Data Files": [
            ("data/fraudshield_cases.json", "Frozen snapshot (108 cases)"),
            ("data/creditcard.csv", "Source data (optional)"),
        ],
        "Documentation": [
            ("README.md", "User documentation"),
            ("server/__init__.py", "Package marker"),
        ],
        "Testing & Validation": [
            ("validate_enhancements.py", "Enhancement tests"),
            ("validate_api.py", "API endpoint tests"),
        ],
    }
    
    all_pass = True
    total_files = 0
    found_files = 0
    
    for category, files in checks.items():
        print(f"📁 {category}")
        print("-" * 80)
        
        for filepath, description in files:
            total_files += 1
            exists = Path(filepath).exists()
            status = "✓" if exists else "✗"
            
            if exists:
                found_files += 1
                try:
                    size = Path(filepath).stat().st_size
                    size_str = f"({size:,} bytes)"
                except:
                    size_str = ""
                print(f"  {status} {filepath:40} {description:35} {size_str}")
            else:
                print(f"  {status} {filepath:40} {description:35} MISSING!")
                all_pass = False
        
        print()
    
    # Check git status
    print("📊 Git Configuration")
    print("-" * 80)
    
    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            cwd="."
        )
        if "github.com/DevikaJ2005/Fraudshield" in result.stdout:
            print("  ✓ GitHub remote configured")
            print("    " + result.stdout.split('\n')[0])
        else:
            print("  ⚠ GitHub remote not found")
    except:
        pass
    
    # Check latest commit
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True,
            text=True,
            cwd="."
        )
        print(f"  ✓ Latest commit: {result.stdout.strip()}")
    except:
        pass
    
    print()
    print("=" * 80)
    print("DEPLOYMENT READINESS SUMMARY")
    print("=" * 80)
    print(f"Files Ready: {found_files}/{total_files}")
    
    if all_pass:
        print("✓ ALL FILES PRESENT - READY FOR DEPLOYMENT")
    else:
        print("✗ SOME FILES MISSING - DEPLOYMENT MAY FAIL")
    
    print()
    print("=" * 80)
    print("NEXT STEPS FOR HF SPACE DEPLOYMENT")
    print("=" * 80)
    print()
    print("1. Go to https://huggingface.co/new-space")
    print()
    print("2. Fill in the form:")
    print("   - Owner: DevikaJ2005 (or your HF username)")
    print("   - Space name: fraudshield")
    print("   - Space SDK: Docker")
    print("   - Visibility: Public")
    print()
    print("3. Choose connection method:")
    print()
    print("   Option A: GitHub Sync (RECOMMENDED)")
    print("   • After creating space, go to Settings")
    print("   • Find 'Repository settings'")
    print("   • Enable GitHub Sync")
    print("   • Select: DevikaJ2005/Fraudshield")
    print("   • Branch: main")
    print("   • Save - Deploy will start automatically!")
    print()
    print("   Option B: Manual Git Push")
    print("   • After creating space, HF shows git URL")
    print("   • Run: git remote add space <HF_GIT_URL>")
    print("   • Run: git push space main")
    print()
    print("4. Wait for deployment (~2-5 minutes)")
    print()
    print("5. After deployment, verify:")
    print("   • GET https://YOUR_SPACE_URL/health → status: healthy")
    print("   • GET https://YOUR_SPACE_URL/info → snapshot metadata")
    print("   • GET https://YOUR_SPACE_URL/tasks → 3 tasks listed")
    print()
    print("=" * 80)
    print()

if __name__ == "__main__":
    check_deployment_readiness()
