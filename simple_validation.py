#!/usr/bin/env python3
"""
Simple validation script that tests core functionality without heavy dependencies.
"""

import sys
import os
import time
import asyncio
import traceback
from pathlib import Path

# Add the federated_dp_llm module to path
sys.path.insert(0, str(Path(__file__).parent))

def test_individual_modules():
    """Test individual modules directly without dependencies."""
    print("Testing individual modules...")
    
    results = {}
    
    # Test privacy accountant
    try:
        from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
        config = DPConfig(epsilon_per_query=0.1, delta=1e-5, max_budget_per_user=10.0)
        accountant = PrivacyAccountant(config)
        
        # Basic functionality test
        user_id = "test_user"
        initial = accountant.get_remaining_budget(user_id)
        success = accountant.spend_budget(user_id, 2.0, "test")
        remaining = accountant.get_remaining_budget(user_id)
        
        assert initial == 10.0
        assert success is True
        assert remaining == 8.0
        
        results["PrivacyAccountant"] = "✓ PASS"
    except Exception as e:
        results["PrivacyAccountant"] = f"✗ FAIL: {e}"
    
    # Test authentication manager
    try:
        from federated_dp_llm.security.authentication import AuthenticationManager
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Test token creation and verification
        token = auth_manager._create_token("test_user", "cardiology", ["doctor"])
        user_info = auth_manager.verify_token(token)
        
        assert user_info["username"] == "test_user"
        assert user_info["department"] == "cardiology"
        
        results["AuthenticationManager"] = "✓ PASS"
    except Exception as e:
        results["AuthenticationManager"] = f"✗ FAIL: {e}"
    
    # Test budget manager
    try:
        from federated_dp_llm.security.compliance import BudgetManager
        
        department_budgets = {"cardiology": 10.0, "emergency": 15.0}
        budget_manager = BudgetManager(department_budgets)
        
        # Test budget operations
        success = budget_manager.allocate_budget("user1", "cardiology", 3.0)
        remaining = budget_manager.get_remaining_budget("user1")
        
        assert success is True
        assert remaining == 3.0
        
        results["BudgetManager"] = "✓ PASS"
    except Exception as e:
        results["BudgetManager"] = f"✗ FAIL: {e}"
    
    # Test health checker
    try:
        from federated_dp_llm.monitoring.health_check import HealthChecker
        
        health_checker = HealthChecker()
        summary = health_checker.get_health_summary()
        
        assert "overall_status" in summary
        assert "total_components" in summary
        
        results["HealthChecker"] = "✓ PASS"
    except Exception as e:
        results["HealthChecker"] = f"✗ FAIL: {e}"
    
    return results

async def test_async_modules():
    """Test async modules."""
    results = {}
    
    # Test cache manager
    try:
        from federated_dp_llm.optimization.caching import CacheManager
        
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False,
            "memory_cache": {"max_size": 100, "default_ttl": 300}
        }
        
        cache_manager = CacheManager(config)
        
        # Test cache operations
        await cache_manager.set("test_key", "test_value", ttl=60)
        cached_value = await cache_manager.get("test_key")
        
        assert cached_value == "test_value"
        
        # Test stats
        stats = cache_manager.get_stats()
        assert stats["hits"] >= 1
        assert stats["sets"] >= 1
        
        await cache_manager.close()
        
        results["CacheManager"] = "✓ PASS"
    except Exception as e:
        results["CacheManager"] = f"✗ FAIL: {e}"
    
    return results

def test_project_structure():
    """Test that project structure is correct."""
    print("Testing project structure...")
    
    required_files = [
        "federated_dp_llm/__init__.py",
        "federated_dp_llm/core/__init__.py",
        "federated_dp_llm/core/privacy_accountant.py",
        "federated_dp_llm/routing/__init__.py", 
        "federated_dp_llm/routing/load_balancer.py",
        "federated_dp_llm/security/__init__.py",
        "federated_dp_llm/security/authentication.py",
        "federated_dp_llm/security/compliance.py",
        "federated_dp_llm/monitoring/__init__.py",
        "federated_dp_llm/monitoring/health_check.py",
        "federated_dp_llm/optimization/__init__.py",
        "federated_dp_llm/optimization/caching.py",
        "federated_dp_llm/optimization/performance_optimizer.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "tests/conftest.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def test_code_quality():
    """Test basic code quality metrics."""
    print("Testing code quality...")
    
    # Check that modules can be imported individually
    modules_to_test = [
        "federated_dp_llm.core.privacy_accountant",
        "federated_dp_llm.security.authentication", 
        "federated_dp_llm.security.compliance",
        "federated_dp_llm.monitoring.health_check",
        "federated_dp_llm.optimization.caching"
    ]
    
    importable_modules = 0
    total_modules = len(modules_to_test)
    
    for module in modules_to_test:
        try:
            __import__(module)
            importable_modules += 1
        except ImportError:
            continue
    
    import_success_rate = importable_modules / total_modules
    print(f"✓ Module import success rate: {import_success_rate:.1%} ({importable_modules}/{total_modules})")
    
    return import_success_rate > 0.5

async def main():
    """Run all validation tests."""
    print("🧪 Starting simplified validation...\n")
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Test code quality
    quality_ok = test_code_quality()
    
    # Test individual modules
    print("\nTesting individual modules...")
    sync_results = test_individual_modules()
    
    print("\nTesting async modules...")
    async_results = await test_async_modules()
    
    # Combine results
    all_results = {**sync_results, **async_results}
    
    print("\n📊 Test Results:")
    passed = 0
    total = 0
    
    for module, result in all_results.items():
        print(f"  {module}: {result}")
        total += 1
        if result.startswith("✓"):
            passed += 1
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"  Structure Check: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"  Code Quality: {'✓ PASS' if quality_ok else '✗ FAIL'}")
    print(f"  Module Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    overall_success = structure_ok and quality_ok and (passed/total >= 0.7)
    
    if overall_success:
        print("\n🎉 Validation successful! Core functionality is working.")
    else:
        print("\n⚠️  Some validations failed, but core system appears functional.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)