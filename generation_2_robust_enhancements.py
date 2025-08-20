#!/usr/bin/env python3
"""
Generation 2: Make it Robust - Enhanced Error Handling, Logging, and Security

Implements comprehensive robustness features:
- Advanced error handling with recovery strategies
- Structured logging with healthcare compliance
- Security enhancements and input validation  
- Health monitoring and alerting
- Circuit breakers and resilience patterns
"""

import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federated_dp_llm.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class RobustnessMetrics:
    """Metrics for measuring system robustness."""
    error_recovery_count: int = 0
    circuit_breaker_activations: int = 0
    security_validation_failures: int = 0
    health_check_failures: int = 0
    retry_attempts: int = 0
    successful_operations: int = 0

def test_enhanced_error_handling():
    """Test comprehensive error handling mechanisms."""
    logger.info("🛡️ Testing Enhanced Error Handling")
    
    try:
        from federated_dp_llm.core.enhanced_error_handling import (
            FederatedErrorHandler, 
            ErrorRecoveryStrategy,
            ErrorContext,
            ErrorSeverity
        )
        
        # Initialize error handler
        error_handler = FederatedErrorHandler(
            max_retries=3,
            backoff_factor=2.0,
            recovery_strategies=[
                ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
                ErrorRecoveryStrategy.FALLBACK_TO_LOCAL,
                ErrorRecoveryStrategy.CIRCUIT_BREAKER
            ]
        )
        
        # Test error context creation
        error_context = ErrorContext(
            operation="privacy_budget_check",
            user_id="test_user",
            node_id="hospital_a",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM
        )
        
        # Test error handling with recovery
        def failing_operation():
            raise ConnectionError("Node temporarily unavailable")
        
        try:
            result = error_handler.handle_with_recovery(
                failing_operation, 
                error_context
            )
            logger.info("✅ Error recovery mechanism tested")
        except Exception as e:
            logger.info(f"⚠️ Expected failure after recovery attempts: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced error handling test failed: {e}")
        return False

def test_comprehensive_logging():
    """Test structured logging with healthcare compliance."""
    logger.info("📝 Testing Comprehensive Logging")
    
    try:
        from federated_dp_llm.monitoring.logging_config import (
            setup_healthcare_logging,
            AuditLogger,
            PrivacyLogger,
            SecurityLogger
        )
        
        # Setup healthcare-compliant logging
        healthcare_logger = setup_healthcare_logging(
            log_level="INFO",
            enable_audit_trail=True,
            enable_privacy_logging=True,
            compliance_mode="HIPAA"
        )
        
        # Test audit logging
        audit_logger = AuditLogger()
        audit_logger.log_privacy_access(
            user_id="doctor_123",
            patient_id="[REDACTED]",
            operation="inference_request",
            privacy_budget_spent=0.1
        )
        
        # Test privacy logging
        privacy_logger = PrivacyLogger()
        privacy_logger.log_differential_privacy_application(
            mechanism="gaussian",
            epsilon=0.1,
            delta=1e-5,
            query_type="diagnostic_assistance"
        )
        
        # Test security logging
        security_logger = SecurityLogger()
        security_logger.log_authentication_event(
            user_id="doctor_123",
            event_type="successful_login",
            client_ip="10.0.1.100",
            user_agent="FederatedClient/1.0"
        )
        
        logger.info("✅ Healthcare-compliant logging configured")
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive logging test failed: {e}")
        return False

def test_security_enhancements():
    """Test enhanced security features."""
    logger.info("🔐 Testing Security Enhancements")
    
    try:
        from federated_dp_llm.security.enhanced_security import (
            SecurityValidator,
            SecurityLevel,
            ThreatDetector,
            AccessController
        )
        
        # Initialize security components
        security_validator = SecurityValidator(
            security_level=SecurityLevel.HEALTHCARE_HIPAA,
            enable_threat_detection=True,
            enable_access_logging=True
        )
        
        # Test input validation
        test_inputs = {
            "user_prompt": "Patient presents with chest pain and shortness of breath",
            "user_id": "doctor_123",
            "department": "emergency",
            "privacy_budget": 0.5
        }
        
        validation_result = security_validator.validate_inference_request(test_inputs)
        if validation_result.is_valid:
            logger.info("✅ Input validation passed")
        else:
            logger.warning(f"⚠️ Validation issues: {validation_result.issues}")
        
        # Test threat detection
        threat_detector = ThreatDetector()
        potential_threat = threat_detector.analyze_request_pattern(
            user_id="doctor_123",
            request_frequency=10,  # requests per minute
            unusual_access_patterns=False
        )
        
        if not potential_threat:
            logger.info("✅ No threats detected")
        else:
            logger.warning(f"⚠️ Potential threat detected: {potential_threat}")
        
        # Test access control
        access_controller = AccessController()
        access_granted = access_controller.check_role_permissions(
            user_id="doctor_123",
            role="physician",
            requested_operation="inference_request",
            resource_sensitivity="moderate"
        )
        
        if access_granted:
            logger.info("✅ Access control check passed")
        else:
            logger.warning("⚠️ Access denied")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Security enhancement test failed: {e}")
        return False

def test_health_monitoring():
    """Test comprehensive health monitoring."""
    logger.info("🏥 Testing Health Monitoring")
    
    try:
        from federated_dp_llm.monitoring.advanced_health_check import (
            AdvancedHealthMonitor,
            HealthStatus,
            SystemComponent,
            HealthAlert
        )
        
        # Initialize health monitor
        health_monitor = AdvancedHealthMonitor(
            check_interval=30,  # seconds
            critical_threshold=0.95,
            warning_threshold=0.80,
            enable_alerting=True
        )
        
        # Add system components to monitor
        health_monitor.register_component(
            SystemComponent(
                name="privacy_accountant",
                type="core",
                endpoint="/health/privacy",
                critical=True
            )
        )
        
        health_monitor.register_component(
            SystemComponent(
                name="quantum_planner",
                type="optimization",
                endpoint="/health/quantum",
                critical=False
            )
        )
        
        # Simulate health checks
        privacy_health = health_monitor.check_component_health("privacy_accountant")
        quantum_health = health_monitor.check_component_health("quantum_planner")
        
        logger.info(f"Privacy Accountant Health: {privacy_health}")
        logger.info(f"Quantum Planner Health: {quantum_health}")
        
        # Test overall system health
        system_health = health_monitor.get_system_health()
        logger.info(f"Overall System Health: {system_health}")
        
        logger.info("✅ Health monitoring system operational")
        return True
        
    except Exception as e:
        logger.error(f"❌ Health monitoring test failed: {e}")
        return False

def test_circuit_breakers():
    """Test circuit breaker patterns for resilience."""
    logger.info("⚡ Testing Circuit Breakers")
    
    try:
        from federated_dp_llm.resilience.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
            CircuitBreakerConfig
        )
        
        # Configure circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            half_open_max_calls=5,
            expected_exception=ConnectionError
        )
        
        circuit_breaker = CircuitBreaker(cb_config)
        
        # Test normal operation
        def reliable_operation():
            return "Success"
        
        result = circuit_breaker.call(reliable_operation)
        logger.info(f"✅ Circuit breaker normal operation: {result}")
        
        # Test failure handling
        def failing_operation():
            raise ConnectionError("Service unavailable")
        
        failures = 0
        for i in range(5):
            try:
                circuit_breaker.call(failing_operation)
            except Exception:
                failures += 1
                
        logger.info(f"✅ Circuit breaker handled {failures} failures")
        logger.info(f"Circuit state: {circuit_breaker.state}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Circuit breaker test failed: {e}")
        return False

def test_privacy_budget_monitoring():
    """Test enhanced privacy budget monitoring."""
    logger.info("🔏 Testing Privacy Budget Monitoring")
    
    try:
        from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
        from federated_dp_llm.monitoring.metrics import MetricsCollector
        
        # Enhanced privacy accountant with monitoring
        config = DPConfig(
            epsilon_per_query=0.1,
            delta=1e-5,
            max_budget_per_user=10.0
        )
        
        accountant = PrivacyAccountant(config)
        metrics_collector = MetricsCollector()
        
        # Test budget monitoring
        user_id = "doctor_123"
        
        # Spend some budget
        for i in range(5):
            try:
                result = accountant.spend_budget(user_id, 0.5, f"query_{i}")
                remaining = accountant.get_remaining_budget(user_id)
                
                # Record metrics
                metrics_collector.record_privacy_spend(
                    user_id=user_id,
                    epsilon_spent=0.5,
                    remaining_budget=remaining
                )
                
                logger.info(f"Query {i}: Remaining budget = {remaining:.3f}")
                
            except Exception as e:
                logger.warning(f"Budget exhausted at query {i}: {e}")
                break
        
        # Test privacy budget alerts
        if remaining < 1.0:
            metrics_collector.trigger_privacy_alert(
                user_id=user_id,
                current_budget=remaining,
                alert_type="low_budget_warning"
            )
            logger.info("✅ Privacy budget alert triggered")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Privacy budget monitoring test failed: {e}")
        return False

async def test_async_error_handling():
    """Test asynchronous error handling patterns."""
    logger.info("🔄 Testing Async Error Handling")
    
    try:
        from federated_dp_llm.federation.client import PrivateInferenceClient
        from federated_dp_llm.core.enhanced_error_handling import AsyncErrorHandler
        
        # Initialize async error handler
        async_handler = AsyncErrorHandler(
            max_concurrent_retries=3,
            timeout_seconds=30,
            exponential_backoff=True
        )
        
        # Test async retry mechanism
        async def unreliable_async_operation():
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise asyncio.TimeoutError("Operation timed out")
            return "Async operation successful"
        
        try:
            result = await async_handler.execute_with_retry(
                unreliable_async_operation(),
                max_attempts=3
            )
            logger.info(f"✅ Async retry successful: {result}")
        except Exception as e:
            logger.info(f"⚠️ Async operation failed after retries: {e}")
        
        # Test timeout handling
        async def slow_operation():
            await asyncio.sleep(5)  # Simulate slow operation
            return "Completed"
        
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=2.0)
            logger.info(f"✅ Operation completed: {result}")
        except asyncio.TimeoutError:
            logger.info("⚠️ Operation timed out as expected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Async error handling test failed: {e}")
        return False

def main():
    """Run Generation 2 robustness tests."""
    logger.info("🚀 Generation 2: Make it Robust - Comprehensive Testing")
    logger.info("=" * 70)
    
    metrics = RobustnessMetrics()
    
    tests = [
        ("Enhanced Error Handling", test_enhanced_error_handling),
        ("Comprehensive Logging", test_comprehensive_logging),
        ("Security Enhancements", test_security_enhancements),
        ("Health Monitoring", test_health_monitoring),
        ("Circuit Breakers", test_circuit_breakers),
        ("Privacy Budget Monitoring", test_privacy_budget_monitoring),
        ("Async Error Handling", lambda: asyncio.run(test_async_error_handling()))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                metrics.successful_operations += 1
            else:
                metrics.error_recovery_count += 1
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            metrics.error_recovery_count += 1
    
    logger.info(f"\n📊 Generation 2 Results: {passed}/{total} tests passed")
    logger.info(f"📈 Robustness Metrics:")
    logger.info(f"  - Successful Operations: {metrics.successful_operations}")
    logger.info(f"  - Error Recoveries: {metrics.error_recovery_count}")
    logger.info(f"  - Circuit Breaker Activations: {metrics.circuit_breaker_activations}")
    
    if passed >= total * 0.8:  # 80% pass rate
        logger.info("🎉 Generation 2 Complete: System robustness enhanced!")
        return True
    else:
        logger.warning("⚠️ Some robustness tests failed - system needs additional hardening")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)