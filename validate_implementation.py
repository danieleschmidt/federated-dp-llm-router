#!/usr/bin/env python3
"""
Basic validation script to test core functionality without pytest.
"""

import sys
import os
import time
import asyncio
import traceback
from pathlib import Path

# Add the federated_dp_llm module to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
        from federated_dp_llm.routing.load_balancer import FederatedRouter, InferenceRequest
        from federated_dp_llm.federation.client import HospitalNode
        from federated_dp_llm.security.authentication import AuthenticationManager
        from federated_dp_llm.security.compliance import BudgetManager
        from federated_dp_llm.monitoring.health_check import HealthChecker
        from federated_dp_llm.optimization.caching import CacheManager
        from federated_dp_llm.optimization.performance_optimizer import PerformanceOptimizer
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_privacy_accountant():
    """Test privacy accountant basic functionality."""
    print("\nTesting PrivacyAccountant...")
    
    try:
        from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
        
        config = DPConfig(epsilon_per_query=0.1, delta=1e-5, max_budget_per_user=10.0)
        accountant = PrivacyAccountant(config)
        
        # Test budget allocation
        user_id = "test_user"
        initial_budget = accountant.get_remaining_budget(user_id)
        assert initial_budget == config.max_budget_per_user
        
        # Test spending budget
        success = accountant.spend_budget(user_id, 2.0, "test_query")
        assert success is True
        
        remaining = accountant.get_remaining_budget(user_id)
        assert remaining == initial_budget - 2.0
        
        print("✓ PrivacyAccountant tests passed")
        return True
    except Exception as e:
        print(f"✗ PrivacyAccountant test failed: {e}")
        traceback.print_exc()
        return False

def test_federated_router():
    """Test federated router basic functionality."""
    print("\nTesting FederatedRouter...")
    
    try:
        from federated_dp_llm.routing.load_balancer import FederatedRouter
        from federated_dp_llm.federation.client import HospitalNode
        
        router = FederatedRouter(model_name="test-model", num_shards=2)
        
        # Create test nodes
        nodes = [
            HospitalNode(
                id="hospital_a",
                endpoint="https://hospital-a.test:8443",
                data_size=50000,
                compute_capacity="4xA100",
                department="cardiology"
            ),
            HospitalNode(
                id="hospital_b",
                endpoint="https://hospital-b.test:8443", 
                data_size=75000,
                compute_capacity="8xA100",
                department="emergency"
            )
        ]
        
        # Register nodes
        router.register_nodes(nodes)
        assert len(router.registered_nodes) == 2
        
        # Test node selection
        best_node = router._select_best_node("cardiology")
        assert best_node in router.registered_nodes
        
        print("✓ FederatedRouter tests passed")
        return True
    except Exception as e:
        print(f"✗ FederatedRouter test failed: {e}")
        traceback.print_exc()
        return False

async def test_cache_manager():
    """Test cache manager basic functionality."""
    print("\nTesting CacheManager...")
    
    try:
        from federated_dp_llm.optimization.caching import CacheManager
        
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False,
            "memory_cache": {
                "max_size": 100,
                "default_ttl": 300
            }
        }
        
        cache_manager = CacheManager(config)
        
        # Test cache operations
        await cache_manager.set("test_key", "test_value", ttl=60)
        
        cached_value = await cache_manager.get("test_key")
        assert cached_value == "test_value"
        
        # Test non-existent key
        missing_value = await cache_manager.get("missing_key")
        assert missing_value is None
        
        # Test stats
        stats = cache_manager.get_stats()
        assert stats["hits"] >= 1
        assert stats["sets"] >= 1
        
        await cache_manager.close()
        
        print("✓ CacheManager tests passed")
        return True
    except Exception as e:
        print(f"✗ CacheManager test failed: {e}")
        traceback.print_exc()
        return False

def test_authentication():
    """Test authentication manager basic functionality."""
    print("\nTesting AuthenticationManager...")
    
    try:
        from federated_dp_llm.security.authentication import AuthenticationManager
        
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Test token creation
        token = auth_manager._create_token("test_user", "cardiology", ["doctor"])
        assert token is not None
        assert isinstance(token, str)
        
        # Test token verification
        user_info = auth_manager.verify_token(token)
        assert user_info["username"] == "test_user"
        assert user_info["department"] == "cardiology"
        
        print("✓ AuthenticationManager tests passed")
        return True
    except Exception as e:
        print(f"✗ AuthenticationManager test failed: {e}")
        traceback.print_exc()
        return False

def test_budget_manager():
    """Test budget manager basic functionality."""
    print("\nTesting BudgetManager...")
    
    try:
        from federated_dp_llm.security.compliance import BudgetManager
        
        department_budgets = {
            "cardiology": 10.0,
            "emergency": 15.0,
            "radiology": 8.0
        }
        
        budget_manager = BudgetManager(department_budgets)
        
        # Test budget allocation
        success = budget_manager.allocate_budget("user1", "cardiology", 3.0)
        assert success is True
        
        remaining = budget_manager.get_remaining_budget("user1")
        assert remaining == 3.0
        
        # Test budget spending
        success = budget_manager.spend_budget("user1", 1.0)
        assert success is True
        
        remaining = budget_manager.get_remaining_budget("user1")
        assert remaining == 2.0
        
        print("✓ BudgetManager tests passed")
        return True
    except Exception as e:
        print(f"✗ BudgetManager test failed: {e}")
        traceback.print_exc()
        return False

async def test_health_checker():
    """Test health checker basic functionality."""
    print("\nTesting HealthChecker...")
    
    try:
        from federated_dp_llm.monitoring.health_check import HealthChecker
        
        health_checker = HealthChecker()
        
        # Test health summary
        summary = health_checker.get_health_summary()
        assert "overall_status" in summary
        assert "total_components" in summary
        
        # Test builtin checks are registered
        assert len(health_checker.components) > 0
        
        print("✓ HealthChecker tests passed")
        return True
    except Exception as e:
        print(f"✗ HealthChecker test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all validation tests."""
    print("🧪 Starting implementation validation...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Privacy Accountant", test_privacy_accountant),
        ("Federated Router", test_federated_router),
        ("Cache Manager", test_cache_manager),
        ("Authentication", test_authentication),
        ("Budget Manager", test_budget_manager),
        ("Health Checker", test_health_checker)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n📊 Validation Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🎯 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All validations passed! Implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} validation(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)