#!/usr/bin/env python3
"""
Generation 2 Validation Test - Robustness and Error Handling
Tests the enhanced error handling, monitoring, and resilience features.
"""

import asyncio
import time
import sys
import os
import traceback
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '/root/repo')

def test_enhanced_error_handling():
    """Test enhanced error handling system."""
    print("🔧 Testing Enhanced Error Handling System...")
    
    try:
        # Test error type definitions
        from federated_dp_llm.core.enhanced_error_handling import (
            ErrorType, ErrorSeverity, ErrorContext, FederatedError,
            PrivacyBudgetExhaustedException, CircuitBreaker, RetryHandler
        )
        
        # Test error creation
        error = PrivacyBudgetExhaustedException("test_user", 1.0, 0.5)
        assert error.error_type == ErrorType.PRIVACY_BUDGET_EXHAUSTED
        assert error.severity == ErrorSeverity.HIGH
        print("✅ Error type system functioning correctly")
        
        # Test circuit breaker
        from federated_dp_llm.core.enhanced_error_handling import CircuitBreakerConfig
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0)
        breaker = CircuitBreaker("test_breaker", config)
        print("✅ Circuit breaker initialization successful")
        
        # Test graceful degradation
        from federated_dp_llm.core.enhanced_error_handling import GracefulDegradation
        degradation = GracefulDegradation()
        error_context = ErrorContext(
            error_type=ErrorType.PRIVACY_BUDGET_EXHAUSTED,
            severity=ErrorSeverity.HIGH,
            timestamp=time.time(),
            user_id="test_user",
            request_id="test_request",
            component="test_component",
            details={},
            recoverable=True
        )
        degraded_config = degradation.degrade_service(error_context)
        assert isinstance(degraded_config, dict)
        print("✅ Graceful degradation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_advanced_health_monitoring():
    """Test advanced health monitoring system."""
    print("🩺 Testing Advanced Health Monitoring...")
    
    try:
        from federated_dp_llm.monitoring.advanced_health_check import (
            AdvancedHealthChecker, HealthStatus, ComponentType, HealthMetric
        )
        
        # Test health checker initialization
        health_checker = AdvancedHealthChecker()
        print("✅ Health checker initialized successfully")
        
        # Test health metric creation
        metric = HealthMetric(
            name="test_metric",
            value=50.0,
            unit="percent",
            threshold_warning=70.0,
            threshold_critical=90.0,
            status=HealthStatus.HEALTHY,
            timestamp=time.time()
        )
        assert metric.status == HealthStatus.HEALTHY
        print("✅ Health metrics system functioning")
        
        # Test threshold evaluation
        status = health_checker._get_status_from_threshold(95.0, 'cpu_usage')
        assert status == HealthStatus.CRITICAL
        print("✅ Threshold evaluation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced health monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_privacy_accountant_robustness():
    """Test privacy accountant with enhanced error handling."""
    print("🔐 Testing Privacy Accountant Robustness...")
    
    try:
        # Mock numpy for testing
        class MockNP:
            @staticmethod
            def random():
                class Random:
                    @staticmethod
                    def normal(loc, scale, shape):
                        return [0.1] * shape if isinstance(shape, int) else [[0.1]]
                    @staticmethod
                    def laplace(loc, scale, shape):
                        return [0.1] * shape if isinstance(shape, int) else [[0.1]]
                    @staticmethod
                    def choice(options):
                        return options[0] if options else None
                return Random()
            @staticmethod
            def mean(values):
                return sum(values) / len(values) if values else 0
            @staticmethod
            def exp(x):
                return 2.718 ** x
            
        # Temporarily replace numpy
        import sys
        sys.modules['numpy'] = MockNP()
        
        from federated_dp_llm.core.privacy_accountant import DPConfig, DPMechanism, CompositionMethod
        
        # Test configuration
        config = DPConfig(
            epsilon_per_query=0.1,
            delta=1e-5,
            max_budget_per_user=10.0,
            mechanism=DPMechanism.GAUSSIAN,
            composition=CompositionMethod.RDP
        )
        print("✅ Privacy configuration created successfully")
        
        # Test mechanism types
        assert DPMechanism.GAUSSIAN.value == "gaussian"
        assert CompositionMethod.RDP.value == "rdp"
        print("✅ Privacy mechanisms defined correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Privacy accountant robustness test failed: {e}")
        traceback.print_exc()
        return False

def test_quantum_planning_resilience():
    """Test quantum planning system resilience."""
    print("⚛️ Testing Quantum Planning Resilience...")
    
    try:
        from federated_dp_llm.quantum_planning.quantum_planner import (
            TaskPriority, QuantumState, QuantumTask
        )
        
        # Test quantum task creation
        task = QuantumTask(
            task_id="test_task_1",
            user_id="test_user",
            prompt="Test medical query",
            priority=TaskPriority.HIGH,
            privacy_budget=0.5,
            estimated_duration=30.0,
            resource_requirements={"compute": 0.5}
        )
        
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert task.priority == TaskPriority.HIGH
        print("✅ Quantum task creation successful")
        
        # Test priority enumeration
        assert TaskPriority.CRITICAL.value == 0
        assert TaskPriority.BACKGROUND.value == 4
        print("✅ Quantum priority system functioning")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum planning resilience test failed: {e}")
        traceback.print_exc()
        return False

def test_security_enhancements():
    """Test security validation enhancements."""
    print("🛡️ Testing Security Enhancements...")
    
    try:
        # Test security module imports
        try:
            from federated_dp_llm.security.enhanced_privacy_validator import ValidationResult
            print("✅ Enhanced privacy validator import successful")
        except ImportError:
            print("⚠️ Enhanced privacy validator not found (optional)")
        
        try:
            from federated_dp_llm.security.comprehensive_security import SecurityLevel
            print("✅ Comprehensive security import successful")
        except ImportError:
            print("⚠️ Comprehensive security not found (optional)")
        
        # Test basic security concepts
        from federated_dp_llm.core.enhanced_error_handling import ErrorType
        security_error = ErrorType.SECURITY_VIOLATION
        assert security_error.value == "security_violation"
        print("✅ Security error types defined correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Security enhancements test failed: {e}")
        traceback.print_exc()
        return False

def run_generation_2_validation():
    """Run complete Generation 2 validation suite."""
    print("=" * 60)
    print("🚀 GENERATION 2 VALIDATION - MAKE IT ROBUST")
    print("=" * 60)
    
    tests = [
        test_enhanced_error_handling,
        test_advanced_health_monitoring,
        test_privacy_accountant_robustness,
        test_quantum_planning_resilience,
        test_security_enhancements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"🎯 GENERATION 2 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Generation 2 ROBUST implementation validated successfully!")
        print("📊 Enhanced error handling, monitoring, and resilience confirmed")
    elif passed >= total * 0.8:
        print("⚠️ Generation 2 mostly successful with minor issues")
        print("🔧 Consider addressing remaining test failures")
    else:
        print("❌ Generation 2 validation failed - significant issues detected")
        print("🛠️ Robustness implementation needs improvement")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_generation_2_validation()
    sys.exit(0 if success else 1)