#!/usr/bin/env python3
"""
Security audit script for DeepStack Trainer
Checks for common security vulnerabilities and best practices
"""

import os
import sys
import subprocess
import json
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


def check_dependencies():
    """Check for vulnerable dependencies"""
    print("🔍 Checking dependencies for security vulnerabilities...")
    
    # Check if safety is installed
    returncode, _, _ = run_command(["safety", "--version"])
    if returncode != 0:
        print("❌ Safety not installed. Install with: pip install safety")
        return False
    
    # Run safety check
    returncode, stdout, stderr = run_command(["safety", "check", "--json"])
    if returncode == 0:
        print("✅ No known security vulnerabilities found in dependencies")
        return True
    else:
        try:
            vulnerabilities = json.loads(stdout)
            if vulnerabilities:
                print(f"⚠️  Found {len(vulnerabilities)} security vulnerabilities:")
                for vuln in vulnerabilities:
                    print(f"   - {vuln.get('package', 'Unknown')}: {vuln.get('advisory', 'No details')}")
                return False
            else:
                print("✅ No known security vulnerabilities found")
                return True
        except json.JSONDecodeError:
            print(f"❌ Error parsing safety output: {stderr}")
            return False


def check_bandit():
    """Run bandit security linter"""
    print("🔍 Running Bandit security analysis...")
    
    # Check if bandit is installed
    returncode, _, _ = run_command(["bandit", "--version"])
    if returncode != 0:
        print("❌ Bandit not installed. Install with: pip install bandit")
        return False
    
    # Run bandit
    returncode, stdout, stderr = run_command([
        "bandit", 
        "-r", ".", 
        "-f", "json",
        "-c", ".bandit"
    ])
    
    if returncode == 0:
        print("✅ No security issues found by Bandit")
        return True
    else:
        try:
            results = json.loads(stdout)
            issues = results.get('results', [])
            if issues:
                print(f"⚠️  Found {len(issues)} potential security issues:")
                for issue in issues:
                    print(f"   - {issue.get('filename', 'Unknown')}:{issue.get('line_number', '?')} - {issue.get('issue_text', 'No details')}")
                return False
            else:
                print("✅ No security issues found")
                return True
        except json.JSONDecodeError:
            print(f"❌ Error parsing bandit output: {stderr}")
            return False


def check_file_permissions():
    """Check for overly permissive file permissions"""
    print("🔍 Checking file permissions...")
    
    issues = []
    for root, dirs, files in os.walk("."):
        # Skip virtual environments and cache directories
        if any(skip in root for skip in ["myenv", "deepstack-trainer", "__pycache__", ".git", "wandb"]):
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(('.py', '.yaml', '.yml', '.json', '.txt')):
                # Check if file is world-writable
                if os.access(filepath, os.W_OK) and os.stat(filepath).st_mode & 0o002:
                    issues.append(filepath)
    
    if issues:
        print(f"⚠️  Found {len(issues)} files with overly permissive permissions:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ File permissions look good")
        return True


def check_hardcoded_secrets():
    """Check for hardcoded secrets"""
    print("🔍 Checking for hardcoded secrets...")
    
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'key\s*=\s*["\'][A-Za-z0-9+/]{20,}["\']',  # Base64-like keys
    ]
    
    issues = []
    for root, dirs, files in os.walk("."):
        # Skip virtual environments and cache directories
        if any(skip in root for skip in ["myenv", "deepstack-trainer", "__pycache__", ".git", "wandb"]):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for i, line in enumerate(content.split('\n'), 1):
                            for pattern in secret_patterns:
                                import re
                                if re.search(pattern, line, re.IGNORECASE):
                                    issues.append(f"{filepath}:{i} - {line.strip()}")
                except Exception:
                    continue
    
    if issues:
        print(f"⚠️  Found {len(issues)} potential hardcoded secrets:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
        return False
    else:
        print("✅ No obvious hardcoded secrets found")
        return True


def main():
    """Run all security checks"""
    print("🛡️  DeepStack Trainer Security Audit")
    print("=" * 50)
    
    checks = [
        check_dependencies,
        check_bandit,
        check_file_permissions,
        check_hardcoded_secrets,
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
    print(f"📊 Security Audit Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All security checks passed!")
        return 0
    else:
        print("⚠️  Some security issues were found. Please review and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
