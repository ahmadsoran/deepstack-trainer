#!/usr/bin/env python3
"""
Code quality check script for DeepStack Trainer
Runs various linting and formatting tools
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any


def run_command(cmd: List[str], cwd: str = None) -> tuple:
    """Run a command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_black():
    """Check code formatting with Black"""
    print("🎨 Checking code formatting with Black...")
    
    returncode, _, _ = run_command(["black", "--version"])
    if returncode != 0:
        print("❌ Black not installed. Install with: pip install black")
        return False
    
    # Check formatting
    returncode, stdout, stderr = run_command([
        "black", 
        "--check", 
        "--diff",
        "."
    ])
    
    if returncode == 0:
        print("✅ Code formatting is correct")
        return True
    else:
        print("⚠️  Code formatting issues found:")
        print(stdout)
        print("Run 'black .' to fix formatting issues")
        return False


def check_flake8():
    """Check code style with Flake8"""
    print("🔍 Checking code style with Flake8...")
    
    returncode, _, _ = run_command(["flake8", "--version"])
    if returncode != 0:
        print("❌ Flake8 not installed. Install with: pip install flake8")
        return False
    
    # Run flake8
    returncode, stdout, stderr = run_command(["flake8", "."])
    
    if returncode == 0:
        print("✅ No style issues found")
        return True
    else:
        print("⚠️  Style issues found:")
        print(stdout)
        return False


def check_mypy():
    """Check type hints with MyPy"""
    print("🔍 Checking type hints with MyPy...")
    
    returncode, _, _ = run_command(["mypy", "--version"])
    if returncode != 0:
        print("❌ MyPy not installed. Install with: pip install mypy")
        return False
    
    # Run mypy
    returncode, stdout, stderr = run_command(["mypy", "."])
    
    if returncode == 0:
        print("✅ No type issues found")
        return True
    else:
        print("⚠️  Type issues found:")
        print(stdout)
        return False


def check_isort():
    """Check import sorting with isort"""
    print("📦 Checking import sorting with isort...")
    
    returncode, _, _ = run_command(["isort", "--version"])
    if returncode != 0:
        print("❌ isort not installed. Install with: pip install isort")
        return False
    
    # Check import sorting
    returncode, stdout, stderr = run_command([
        "isort", 
        "--check-only", 
        "--diff",
        "."
    ])
    
    if returncode == 0:
        print("✅ Import sorting is correct")
        return True
    else:
        print("⚠️  Import sorting issues found:")
        print(stdout)
        print("Run 'isort .' to fix import sorting")
        return False


def check_pytest():
    """Run tests with pytest"""
    print("🧪 Running tests with pytest...")
    
    returncode, _, _ = run_command(["pytest", "--version"])
    if returncode != 0:
        print("❌ pytest not installed. Install with: pip install pytest")
        return False
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("⚠️  No tests directory found. Consider adding tests.")
        return True
    
    # Run tests
    returncode, stdout, stderr = run_command(["pytest", "-v"])
    
    if returncode == 0:
        print("✅ All tests passed")
        return True
    else:
        print("⚠️  Some tests failed:")
        print(stdout)
        return False


def check_coverage():
    """Check test coverage"""
    print("📊 Checking test coverage...")
    
    returncode, _, _ = run_command(["coverage", "--version"])
    if returncode != 0:
        print("❌ coverage not installed. Install with: pip install coverage")
        return False
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("⚠️  No tests directory found. Skipping coverage check.")
        return True
    
    # Run coverage
    returncode, stdout, stderr = run_command([
        "coverage", 
        "run", 
        "-m", 
        "pytest"
    ])
    
    if returncode != 0:
        print("⚠️  Tests failed during coverage run:")
        print(stdout)
        return False
    
    # Generate coverage report
    returncode, stdout, stderr = run_command(["coverage", "report"])
    
    if returncode == 0:
        print("📊 Coverage report:")
        print(stdout)
        return True
    else:
        print("❌ Error generating coverage report:")
        print(stderr)
        return False


def main():
    """Run all quality checks"""
    print("🔧 DeepStack Trainer Code Quality Check")
    print("=" * 50)
    
    checks = [
        check_black,
        check_flake8,
        check_mypy,
        check_isort,
        check_pytest,
        check_coverage,
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"❌ Error running {check.__name__}: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Quality Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All quality checks passed!")
        return 0
    else:
        print("⚠️  Some quality issues were found. Please review and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
