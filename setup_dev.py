#!/usr/bin/env python3
"""
Development setup script for DeepStack Trainer
Sets up the development environment with all necessary tools
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and optionally check for errors"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_python_packages():
    """Install Python packages for development"""
    print("📦 Installing Python development packages...")
    
    packages = [
        "black>=23.0.0",
        "flake8>=6.0.0", 
        "mypy>=1.5.0",
        "isort>=5.12.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "bandit>=1.7.0",
        "safety>=2.3.0",
        "pre-commit>=3.5.0",
        "coverage>=7.3.0",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        if not run_command([sys.executable, "-m", "pip", "install", package]):
            print(f"Failed to install {package}")
            return False
    
    return True


def setup_pre_commit():
    """Set up pre-commit hooks"""
    print("🔧 Setting up pre-commit hooks...")
    
    if not run_command(["pre-commit", "install"]):
        print("Failed to install pre-commit hooks")
        return False
    
    if not run_command(["pre-commit", "install", "--hook-type", "pre-push"]):
        print("Failed to install pre-push hooks")
        return False
    
    return True


def create_gitignore_additions():
    """Add development-related entries to .gitignore"""
    print("📝 Updating .gitignore...")
    
    gitignore_additions = [
        "",
        "# Development tools",
        ".pytest_cache/",
        ".mypy_cache/",
        ".coverage",
        "htmlcov/",
        ".tox/",
        ".venv/",
        "venv/",
        "ENV/",
        "env/",
        ".env",
        "",
        "# IDE",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
        "",
        "# Security",
        "*.key",
        "*.pem",
        "*.p12",
        "*.pfx",
        "",
        "# Logs",
        "*.log",
        "logs/",
    ]
    
    gitignore_path = Path(".gitignore")
    
    # Read existing .gitignore
    existing_content = ""
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    
    # Add new entries if they don't exist
    new_entries = []
    for entry in gitignore_additions:
        if entry not in existing_content:
            new_entries.append(entry)
    
    if new_entries:
        with open(gitignore_path, 'a') as f:
            f.write('\n'.join(new_entries))
        print("Added development entries to .gitignore")
    else:
        print(".gitignore already contains development entries")
    
    return True


def create_tests_directory():
    """Create a basic tests directory structure"""
    print("🧪 Creating tests directory...")
    
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_file = tests_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Tests for DeepStack Trainer"""\n')
    
    # Create a basic test file
    test_file = tests_dir / "test_basic.py"
    if not test_file.exists():
        test_content = '''"""Basic tests for DeepStack Trainer"""

import pytest
import torch
import numpy as np


def test_torch_import():
    """Test that PyTorch can be imported"""
    assert torch is not None
    assert torch.__version__ is not None


def test_numpy_import():
    """Test that NumPy can be imported"""
    assert np is not None
    assert np.__version__ is not None


def test_basic_tensor_operations():
    """Test basic tensor operations"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = x + y
    assert torch.equal(z, torch.tensor([5, 7, 9]))


def test_device_selection():
    """Test device selection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device is not None
    assert device.type in ['cpu', 'cuda']


if __name__ == "__main__":
    pytest.main([__file__])
'''
        test_file.write_text(test_content)
    
    return True


def run_initial_checks():
    """Run initial quality and security checks"""
    print("🔍 Running initial quality and security checks...")
    
    # Make scripts executable
    if platform.system() != "Windows":
        os.chmod("security_audit.py", 0o755)
        os.chmod("quality_check.py", 0o755)
        os.chmod("setup_dev.py", 0o755)
    
    # Run security audit
    print("Running security audit...")
    if not run_command([sys.executable, "security_audit.py"], check=False):
        print("Security audit found issues - please review")
    
    # Run quality check
    print("Running quality check...")
    if not run_command([sys.executable, "quality_check.py"], check=False):
        print("Quality check found issues - please review")
    
    return True


def main():
    """Main setup function"""
    print("🚀 DeepStack Trainer Development Setup")
    print("=" * 50)
    
    steps = [
        ("Installing Python packages", install_python_packages),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Updating .gitignore", create_gitignore_additions),
        ("Creating tests directory", create_tests_directory),
        ("Running initial checks", run_initial_checks),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            return 1
        print(f"✅ Completed: {step_name}")
    
    print("\n" + "=" * 50)
    print("🎉 Development setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'pre-commit run --all-files' to check all files")
    print("2. Run 'python security_audit.py' to check security")
    print("3. Run 'python quality_check.py' to check code quality")
    print("4. Run 'pytest' to run tests")
    print("5. Start developing with confidence!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
