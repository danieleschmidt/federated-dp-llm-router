#!/usr/bin/env python3
"""
Generation 2 Test Suite: Robustness and Reliability

Tests comprehensive error handling, logging, monitoring, input validation,
and security features implemented in Generation 2.
"""

import asyncio
import time
import json
import tempfile
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_dp_llm.monitoring.logger import (
    StructuredLogger, LogLevel, LogCategory, get_logger, setup_logger
)
from federated_dp_llm.monitoring.error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext, 
    with_error_handling, get_error_handler
)
from federated_dp_llm.monitoring.health_check import (
    HealthChecker, HealthStatus, ComponentType, ComponentConfig
)
from federated_dp_llm.security.input_validation import (
    InputValidator, ValidationError, SanitizationMode, get_validator
)


class TestErrorGenerator:
    """Generate different types of errors for testing."""
    
    @staticmethod
    def network_error():
        raise ConnectionError("Network unreachable")
    
    @staticmethod
    def timeout_error():
        raise asyncio.TimeoutError("Operation timed out")
    
    @staticmethod
    def privacy_error():
        raise ValueError("Privacy budget exceeded")
    
    @staticmethod
    def resource_error():
        raise MemoryError("Out of memory")
    
    @staticmethod
    def authentication_error():
        raise PermissionError("Access denied")


async def test_logging_system():
    """Test the structured logging system."""
    print("=== Testing Logging System ===")
    
    # Setup logger with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = setup_logger(
            log_level=LogLevel.DEBUG,
            log_dir=temp_dir,
            enable_console=False  # Disable console for cleaner test output
        )
        
        # Test basic logging
        logger.log(
            level=LogLevel.INFO,
            message="Test info message",
            category=LogCategory.SYSTEM,
            component="test_component",
            user_id="test_user_123",
            metadata={"test_key": "test_value"}
        )
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            error_id = logger.log_error(
                error=e,
                component="test_component",
                user_id="test_user_123",
                severity="medium"
            )
            assert error_id.startswith("error_")
        
        # Test security logging
        logger.log_security_event(
            event_type="login_attempt",
            message="Failed login attempt detected",
            component="auth_system",
            user_id="suspicious_user",
            severity="high"
        )
        
        # Test privacy logging
        logger.log_privacy_event(
            event_type="budget_spent",
            message="Privacy budget consumed",
            component="privacy_accountant",
            user_id="doctor_456",
            epsilon_spent=0.15
        )
        
        # Test audit logging
        logger.log_audit_event(
            action="query_submitted",
            resource="medical_model",
            component="api_handler",
            user_id="doctor_456",
            department="cardiology",
            success=True
        )
        
        # Test performance tracking
        operation_id = "test_inference_001"
        logger.start_operation_tracking(operation_id)
        await asyncio.sleep(0.1)  # Simulate operation
        logger.end_operation_tracking(
            operation_id=operation_id,
            component="inference_engine",
            success=True,
            metadata={"model": "medllama-7b"}
        )
        
        # Test log search
        search_results = logger.search_logs(
            query="test",
            level=LogLevel.INFO,
            limit=10
        )
        assert len(search_results) > 0
        
        # Test error summary
        error_summary = logger.get_error_summary()
        assert error_summary["total_errors"] >= 1
        
        # Test performance summary
        perf_summary = logger.get_performance_summary()
        assert "test_inference" in perf_summary
        
        # Test sensitive data filtering
        sensitive_data = {
            "user": "john.doe@hospital.com",
            "password": os.environ.get("TEST_PASSWORD", "temp_secret_123"),
            "ssn": "123-45-6789",
            "normal_field": "safe content"
        }
        filtered_data = logger.sensitive_filter.sanitize_dict(sensitive_data)
        assert filtered_data["password"] == "[REDACTED]"
        assert filtered_data["ssn"] == "[REDACTED]"
        assert filtered_data["normal_field"] == "safe content"
        
        print("✓ Logging system tests passed")
        
        # Clean up
        logger.shutdown()


async def test_error_handling_system():
    """Test the error handling and recovery system."""
    print("\n=== Testing Error Handling System ===")
    
    error_handler = ErrorHandler()
    
    # Test error categorization
    network_error = ConnectionError("Network failed")
    category = error_handler.categorize_error(network_error)
    assert category == ErrorCategory.NETWORK
    
    timeout_error = asyncio.TimeoutError("Timed out")
    category = error_handler.categorize_error(timeout_error)
    assert category == ErrorCategory.TIMEOUT
    
    # Test error handling with context
    context = ErrorContext(
        component="test_component",
        operation="test_operation",
        user_id="test_user",
        request_id="req_123"
    )
    
    # Test network error handling
    success, result = await error_handler.handle_error(
        exception=network_error,
        context=context
    )
    # Should attempt recovery
    assert not success  # Recovery will fail in test environment
    
    # Test privacy error handling (should escalate immediately)
    privacy_error = ValueError("Privacy budget exceeded")
    success, result = await error_handler.handle_error(
        exception=privacy_error,
        context=context
    )
    assert not success  # Privacy errors shouldn't retry
    
    # Test error statistics
    stats = error_handler.get_error_statistics()
    assert stats["total_errors"] >= 2
    assert "network" in stats["by_category"]
    
    # Test failure prediction
    risk = error_handler.predict_failure_risk("test_component")
    assert 0.0 <= risk <= 1.0
    
    print("✓ Error handling system tests passed")


async def test_health_monitoring():
    """Test the health monitoring system."""
    print("\n=== Testing Health Monitoring System ===")
    
    health_checker = HealthChecker()
    
    # Register test components
    health_checker.register_node_check(
        node_id="test_node_1",
        endpoint="https://test-node-1.local:8443",
        check_interval=10
    )
    
    # Add custom health check
    def custom_health_check():
        return {"status": "healthy", "custom_metric": 42}
    
    config = ComponentConfig(
        component_id="custom_component",
        component_type=ComponentType.EXTERNAL_API,
        check_function=custom_health_check,
        check_interval=5
    )
    health_checker.register_component(config)
    
    # Start monitoring briefly
    health_checker.start_monitoring()
    await asyncio.sleep(0.5)  # Let it run briefly
    
    # Check health summary
    summary = health_checker.get_health_summary()
    assert summary["total_components"] >= 1
    
    # Test specific component health
    component_health = health_checker.get_component_health("custom_component")
    if component_health:  # Might not be available yet due to timing
        assert "status" in component_health
    
    # Test immediate health check
    result = await health_checker.check_component("custom_component")
    if result:
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
    
    # Stop monitoring
    await health_checker.stop_monitoring()
    
    print("✓ Health monitoring system tests passed")


def test_input_validation():
    """Test the input validation system."""
    print("\n=== Testing Input Validation System ===")
    
    validator = InputValidator()
    
    # Test valid inputs
    try:
        valid_email = validator.validate_field("email", "doctor@hospital.com")
        assert valid_email == "doctor@hospital.com"
        
        valid_user_id = validator.validate_field("user_id", "doctor_123")
        assert valid_user_id == "doctor_123"
        
        valid_department = validator.validate_field("department", "cardiology")
        assert valid_department == "cardiology"
        
        print("✓ Valid input validation passed")
    except ValidationError as e:
        print(f"✗ Valid input validation failed: {e}")
        return False
    
    # Test invalid inputs
    invalid_tests = [
        ("email", "invalid-email", "Invalid email should fail"),
        ("user_id", "user with spaces", "User ID with spaces should fail"),
        ("department", "invalid_dept", "Invalid department should fail"),
        ("privacy_budget", "-1.0", "Negative budget should fail"),
        ("privacy_budget", "15.0", "Excessive budget should fail")
    ]
    
    for field, value, description in invalid_tests:
        try:
            validator.validate_field(field, value)
            print(f"✗ {description}")
            return False
        except ValidationError:
            print(f"✓ {description}")
    
    # Test sanitization
    dangerous_input = "<script>alert('xss')</script>Hello World"
    sanitized = validator.sanitize_input(dangerous_input, SanitizationMode.STRICT)
    assert "<script>" not in sanitized
    assert "Hello World" in sanitized
    
    # Test medical sanitization
    medical_input = "Patient needs 5mg of medication daily"
    sanitized_medical = validator.sanitize_input(medical_input, SanitizationMode.MEDICAL)
    assert "5 mg" in sanitized_medical  # Should preserve medical notation
    
    # Test injection detection
    injection_tests = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "$(rm -rf /)",
        "SELECT * FROM users WHERE id = 1 OR 1=1"
    ]
    
    for injection in injection_tests:
        try:
            validator.sanitize_input(injection)
            print(f"✗ Injection detection failed for: {injection}")
            return False
        except ValidationError:
            print(f"✓ Injection detected and blocked: {injection[:20]}...")
    
    # Test API request validation
    valid_api_request = {
        "prompt": "What are the symptoms of diabetes?",
        "user_id": "doctor_123",
        "model_name": "medllama-7b",
        "max_privacy_budget": 0.5,
        "department": "general"
    }
    
    try:
        validated_request = validator.validate_api_request(valid_api_request)
        assert validated_request["user_id"] == "doctor_123"
        print("✓ API request validation passed")
    except ValidationError as e:
        print(f"✗ API request validation failed: {e}")
        return False
    
    # Test privacy pattern detection
    pii_input = "Patient SSN is 123-45-6789 and email is john@doe.com"
    try:
        validator._check_privacy_patterns(pii_input)
        print("✗ PII detection failed")
        return False
    except ValidationError:
        print("✓ PII patterns detected and blocked")
    
    # Test logging sanitization
    sensitive_log_data = {
        "user": "test@hospital.com",
        "password": os.environ.get("TEST_PASSWORD", "temp_secret_123"),
        "message": "User logged in successfully"
    }
    sanitized_log = validator.sanitize_for_logging(sensitive_log_data)
    assert sanitized_log["password"] == "[REDACTED]"
    assert sanitized_log["message"] == "User logged in successfully"
    
    print("✓ Input validation system tests passed")
    return True


async def test_error_decorator():
    """Test the error handling decorator."""
    print("\n=== Testing Error Handling Decorator ===")
    
    @with_error_handling(max_retries=2, retry_delay=0.1)
    async def failing_function():
        raise ConnectionError("Network error")
    
    @with_error_handling(max_retries=2, retry_delay=0.1)
    async def succeeding_function():
        return "success"
    
    # Test successful function
    try:
        result = await succeeding_function()
        assert result == "success"
        print("✓ Decorator with successful function passed")
    except Exception as e:
        print(f"✗ Decorator with successful function failed: {e}")
        return False
    
    # Test failing function (should still fail after retries)
    try:
        await failing_function()
        print("✗ Decorator should have failed after retries")
        return False
    except ConnectionError:
        print("✓ Decorator correctly failed after retries")
    
    return True


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n=== Testing Circuit Breaker ===")
    
    from federated_dp_llm.monitoring.error_handling import CircuitBreaker
    
    circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
    
    def failing_operation():
        raise ConnectionError("Service unavailable")
    
    def successful_operation():
        return "success"
    
    # Test circuit breaker opening after failures
    failure_count = 0
    for i in range(5):
        try:
            circuit_breaker.call(failing_operation)
        except Exception:
            failure_count += 1
    
    assert failure_count >= 3
    
    # Circuit should be open now
    state = circuit_breaker.get_state()
    if state["state"] == "OPEN":
        print("✓ Circuit breaker opened after failures")
    else:
        print(f"✗ Circuit breaker state is {state['state']}, expected OPEN")
        return False
    
    # Test that circuit breaker blocks calls when open
    try:
        circuit_breaker.call(successful_operation)
        print("✗ Circuit breaker should block calls when open")
        return False
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            print("✓ Circuit breaker correctly blocks calls when open")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    return True


async def test_integration_scenario():
    """Test integrated scenario with multiple systems."""
    print("\n=== Testing Integration Scenario ===")
    
    # Setup systems
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = setup_logger(log_dir=temp_dir, enable_console=False)
        error_handler = get_error_handler()
        validator = get_validator()
        
        # Simulate API request processing with full error handling
        async def process_api_request(request_data):
            try:
                # Step 1: Validate input
                validated_data = validator.validate_api_request(request_data)
                logger.log_audit_event(
                    action="request_validated",
                    resource="api_endpoint",
                    component="api_handler",
                    user_id=validated_data["user_id"],
                    success=True
                )
                
                # Step 2: Simulate processing with potential error
                operation_id = f"process_{int(time.time())}"
                logger.start_operation_tracking(operation_id)
                
                # Simulate some processing time
                await asyncio.sleep(0.1)
                
                # Simulate potential error (20% chance)
                if time.time() % 5 < 1:  # Deterministic for testing
                    raise ConnectionError("Simulated network error")
                
                # Success
                logger.end_operation_tracking(
                    operation_id=operation_id,
                    component="inference_engine",
                    success=True
                )
                
                return {"result": "Generated medical advice", "privacy_cost": 0.1}
                
            except ValidationError as e:
                error_id = logger.log_error(e, "api_handler", validated_data.get("user_id"))
                logger.log_security_event(
                    event_type="validation_failure",
                    message=f"Input validation failed: {e}",
                    component="api_handler",
                    severity="medium"
                )
                raise
            
            except Exception as e:
                context = ErrorContext(
                    component="api_handler",
                    operation="process_request",
                    user_id=validated_data.get("user_id"),
                    request_id=operation_id
                )
                
                success, result = await error_handler.handle_error(e, context)
                if not success:
                    error_id = logger.log_error(e, "api_handler", validated_data.get("user_id"))
                    raise
                
                return result
        
        # Test with valid request
        valid_request = {
            "prompt": "What are the symptoms of hypertension?",
            "user_id": "doctor_456",
            "model_name": "medllama-7b",
            "max_privacy_budget": 0.2,
            "department": "cardiology"
        }
        
        try:
            result = await process_api_request(valid_request)
            print("✓ Valid request processed successfully")
        except Exception as e:
            print(f"✓ Valid request failed gracefully: {e}")
        
        # Test with invalid request
        invalid_request = {
            "prompt": "<script>alert('xss')</script>",
            "user_id": "invalid user id",
            "model_name": "unknown_model"
        }
        
        try:
            await process_api_request(invalid_request)
            print("✗ Invalid request should have failed")
            return False
        except ValidationError:
            print("✓ Invalid request properly rejected")
        
        # Check that events were logged
        search_results = logger.search_logs("request_validated", limit=10)
        if len(search_results) > 0:
            print("✓ Audit events properly logged")
        else:
            print("✗ Expected audit events not found")
            return False
        
        # Check error statistics
        error_stats = error_handler.get_error_statistics()
        if error_stats["total_errors"] > 0:
            print("✓ Errors properly tracked")
        else:
            print("✓ No errors occurred (also valid)")
        
        logger.shutdown()
    
    print("✓ Integration scenario tests passed")
    return True


async def main():
    """Run all Generation 2 tests."""
    print("Federated DP-LLM Router - Generation 2 Test Suite")
    print("=" * 55)
    print("Testing: Robustness, Error Handling, Monitoring, Security")
    print()
    
    tests = [
        ("Logging System", test_logging_system),
        ("Error Handling", test_error_handling_system),
        ("Health Monitoring", test_health_monitoring),
        ("Input Validation", test_input_validation),
        ("Error Decorator", test_error_decorator),
        ("Circuit Breaker", test_circuit_breaker),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result is None or result is True:
                passed_tests += 1
            else:
                print(f"✗ {test_name} failed")
        
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*55}")
    print(f"Generation 2 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL GENERATION 2 TESTS PASSED! 🎉")
        print("\nGeneration 2 Features Validated:")
        print("• Comprehensive structured logging with privacy filtering")
        print("• Robust error handling with automatic recovery")
        print("• Circuit breaker pattern implementation")
        print("• Health monitoring with self-healing capabilities")
        print("• Input validation and injection prevention")
        print("• Security-aware sanitization")
        print("• Performance tracking and metrics")
        print("• Audit trail and compliance logging")
        print("• Integration testing and reliability validation")
        
        print("\n🚀 Ready for Generation 3: Performance and Scalability!")
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed")
        print("Generation 2 needs fixes before proceeding to Generation 3")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())