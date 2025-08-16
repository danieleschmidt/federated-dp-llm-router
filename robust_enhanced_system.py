#!/usr/bin/env python3
"""
Robust Enhanced System - Generation 2: MAKE IT ROBUST
Comprehensive error handling, validation, security, and resilience patterns.
"""

import asyncio
import time
import uuid
import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import secrets
from contextlib import asynccontextmanager

from federated_dp_llm import (
    PrivacyAccountant, DPConfig, FederatedRouter, 
    HospitalNode, PrivateInferenceClient, BudgetManager
)
from federated_dp_llm.routing.load_balancer import InferenceRequest, InferenceResponse
from federated_dp_llm.security.comprehensive_security import SecurityValidator
from federated_dp_llm.resilience.circuit_breaker import CircuitBreaker
from federated_dp_llm.monitoring.advanced_health_check import AdvancedHealthChecker


class ErrorSeverity(Enum):
    """Error severity levels for proper handling and logging."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    component: str
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed feedback."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 1.0


@dataclass
class RobustSystemConfig:
    """Enhanced configuration with security and resilience parameters."""
    max_concurrent_requests: int = 100
    privacy_budget_per_user: float = 10.0
    epsilon_per_query: float = 0.1
    delta: float = 1e-5
    
    # Security settings
    enable_security_validation: bool = True
    enable_audit_logging: bool = True
    max_request_size_mb: int = 10
    request_timeout_seconds: int = 30
    
    # Resilience settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_enabled: bool = True
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: float = 60.0
    
    # Health monitoring
    health_check_interval: float = 30.0
    performance_monitoring_enabled: bool = True
    
    # Compliance
    hipaa_compliance_mode: bool = True
    gdpr_compliance_mode: bool = True
    audit_retention_days: int = 2555  # 7 years for HIPAA


class RobustErrorHandler:
    """Comprehensive error handling with categorization and recovery."""
    
    def __init__(self, config: RobustSystemConfig):
        self.config = config
        self.error_history: List[ErrorContext] = []
        self.error_stats: Dict[str, int] = {}
        self.logger = logging.getLogger(f"{__name__}.ErrorHandler")
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle errors with comprehensive logging and context capture."""
        
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=severity,
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc() if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] else None,
            user_id=user_id,
            request_id=request_id,
            additional_context=additional_context or {}
        )
        
        # Store error for analysis
        self.error_history.append(error_context)
        self.error_stats[error_context.error_type] = self.error_stats.get(error_context.error_type, 0) + 1
        
        # Log based on severity
        log_message = f"[{error_context.error_id}] {component}: {error_context.message}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_context": error_context})
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={"error_context": error_context})
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"error_context": error_context})
        else:
            self.logger.info(log_message, extra={"error_context": error_context})
        
        return error_context
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(
                1 for e in self.error_history if e.severity == severity
            )
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": dict(self.error_stats),
            "severity_distribution": severity_counts,
            "error_rate": len(recent_errors) / 60.0  # errors per minute
        }


class InputValidator:
    """Comprehensive input validation for security and data integrity."""
    
    def __init__(self, config: RobustSystemConfig):
        self.config = config
        self.security_validator = SecurityValidator() if config.enable_security_validation else None
    
    def validate_clinical_request(
        self,
        user_id: str,
        prompt: str,
        department: str,
        privacy_budget: float,
        **kwargs
    ) -> ValidationResult:
        """Validate clinical inference request comprehensively."""
        
        result = ValidationResult(is_valid=True)
        
        # Basic input validation
        if not user_id or len(user_id.strip()) == 0:
            result.errors.append("User ID is required and cannot be empty")
            result.is_valid = False
        
        if not prompt or len(prompt.strip()) == 0:
            result.errors.append("Clinical prompt is required and cannot be empty")
            result.is_valid = False
        
        # Length validation
        if len(prompt) > 10000:  # 10K character limit
            result.errors.append("Clinical prompt exceeds maximum length (10,000 characters)")
            result.is_valid = False
        
        if len(user_id) > 100:
            result.errors.append("User ID exceeds maximum length (100 characters)")
            result.is_valid = False
        
        # Privacy budget validation
        if privacy_budget <= 0:
            result.errors.append("Privacy budget must be positive")
            result.is_valid = False
        
        if privacy_budget > 1.0:
            result.warnings.append("High privacy budget requested - consider if necessary")
        
        # Department validation
        valid_departments = {
            "emergency", "critical_care", "radiology", "general", "research",
            "cardiology", "oncology", "neurology", "surgery", "pediatrics"
        }
        if department not in valid_departments:
            result.warnings.append(f"Department '{department}' not in standard list")
        
        # Security validation
        if self.security_validator:
            security_result = self.security_validator.validate_input(prompt)
            if not security_result.is_safe:
                result.security_issues.extend(security_result.threats_detected)
                result.is_valid = False
        
        # Content validation
        suspicious_patterns = [
            "DROP TABLE", "SELECT *", "DELETE FROM", "<script>", "javascript:",
            "eval(", "exec(", "import os", "subprocess"
        ]
        
        prompt_upper = prompt.upper()
        for pattern in suspicious_patterns:
            if pattern.upper() in prompt_upper:
                result.security_issues.append(f"Suspicious pattern detected: {pattern}")
                result.is_valid = False
        
        # PHI detection (basic patterns)
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{10,}\b',  # Long numbers (could be IDs)
        ]
        
        import re
        for pattern in phi_patterns:
            if re.search(pattern, prompt):
                result.warnings.append("Potential PHI detected - ensure data is de-identified")
        
        # Calculate confidence score
        error_count = len(result.errors) + len(result.security_issues)
        warning_count = len(result.warnings)
        result.confidence_score = max(0.0, 1.0 - (error_count * 0.3 + warning_count * 0.1))
        
        return result


class RateLimiter:
    """Token bucket rate limiter for request throttling."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0
        self.bucket_size = requests_per_minute
        self.user_buckets: Dict[str, Dict[str, float]] = {}
    
    async def acquire(self, user_id: str, tokens: int = 1) -> bool:
        """Acquire tokens for rate limiting."""
        current_time = time.time()
        
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {
                "tokens": self.bucket_size,
                "last_update": current_time
            }
        
        bucket = self.user_buckets[user_id]
        
        # Add tokens based on elapsed time
        elapsed = current_time - bucket["last_update"]
        bucket["tokens"] = min(
            self.bucket_size,
            bucket["tokens"] + elapsed * self.tokens_per_second
        )
        bucket["last_update"] = current_time
        
        # Check if enough tokens available
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        
        return False


class RobustFederatedSystem:
    """Production-ready federated system with comprehensive robustness."""
    
    def __init__(self, config: RobustSystemConfig):
        self.config = config
        self.system_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.system_state = SystemState.INITIALIZING
        
        # Core components
        self.privacy_config = DPConfig(
            epsilon_per_query=config.epsilon_per_query,
            delta=config.delta,
            max_budget_per_user=config.privacy_budget_per_user
        )
        
        self.privacy_accountant = PrivacyAccountant(self.privacy_config)
        self.budget_manager = BudgetManager({
            "emergency": 20.0,
            "critical_care": 15.0,
            "radiology": 10.0,
            "general": 5.0,
            "research": 2.0
        })
        
        # Robustness components
        self.error_handler = RobustErrorHandler(config)
        self.validator = InputValidator(config)
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute) if config.rate_limit_enabled else None
        
        # Circuit breakers for each major component
        self.circuit_breakers = {}
        if config.circuit_breaker_enabled:
            self.circuit_breakers = {
                "router": CircuitBreaker(
                    failure_threshold=config.circuit_breaker_failure_threshold,
                    recovery_timeout=config.circuit_breaker_recovery_timeout
                ),
                "privacy_accountant": CircuitBreaker(
                    failure_threshold=config.circuit_breaker_failure_threshold,
                    recovery_timeout=config.circuit_breaker_recovery_timeout
                ),
                "validation": CircuitBreaker(
                    failure_threshold=config.circuit_breaker_failure_threshold,
                    recovery_timeout=config.circuit_breaker_recovery_timeout
                )
            }
        
        # Health monitoring
        self.health_checker = AdvancedHealthChecker() if config.performance_monitoring_enabled else None
        
        # Session and audit tracking
        self.active_sessions: Dict[str, Dict] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Async locks for thread safety
        self.session_lock = asyncio.Lock()
        self.audit_lock = asyncio.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.RobustSystem")
        self.logger.info(f"RobustFederatedSystem initialized with ID: {self.system_id}")
    
    async def initialize_with_validation(self, hospitals: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Initialize system with comprehensive validation."""
        try:
            self.system_state = SystemState.INITIALIZING
            validation_errors = []
            
            # Validate hospital configurations
            for i, hospital_config in enumerate(hospitals):
                errors = self._validate_hospital_config(hospital_config)
                if errors:
                    validation_errors.extend([f"Hospital {i+1}: {error}" for error in errors])
            
            if validation_errors:
                self.system_state = SystemState.CRITICAL
                return False, validation_errors
            
            # Initialize core router
            self.router = FederatedRouter(
                model_name="medllama-7b",
                num_shards=min(4, len(hospitals))
            )
            
            # Register hospital nodes
            hospital_nodes = []
            for hospital_config in hospitals:
                node = HospitalNode(
                    id=hospital_config["id"],
                    endpoint=hospital_config["endpoint"],
                    data_size=hospital_config.get("data_size", 50000),
                    compute_capacity=hospital_config.get("compute_capacity", "4xA100"),
                    department=hospital_config.get("department"),
                    region=hospital_config.get("region")
                )
                hospital_nodes.append(node)
            
            await self.router.register_nodes(hospital_nodes)
            
            # Initialize performance tracking
            self._initialize_performance_metrics(hospital_nodes)
            
            # Start health monitoring
            if self.health_checker:
                await self.health_checker.start_monitoring(hospital_nodes)
            
            self.system_state = SystemState.HEALTHY
            self.logger.info(f"System initialized successfully with {len(hospital_nodes)} nodes")
            
            return True, []
            
        except Exception as e:
            error_context = self.error_handler.handle_error(
                e, "system_initialization", ErrorSeverity.CRITICAL
            )
            self.system_state = SystemState.CRITICAL
            return False, [f"Initialization failed: {error_context.message}"]
    
    def _validate_hospital_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate individual hospital configuration."""
        errors = []
        
        required_fields = ["id", "endpoint"]
        for field in required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")
        
        if "endpoint" in config:
            endpoint = config["endpoint"]
            if not endpoint.startswith(("https://", "http://")):
                errors.append("Endpoint must start with https:// or http://")
            
            if self.config.hipaa_compliance_mode and not endpoint.startswith("https://"):
                errors.append("HIPAA compliance requires HTTPS endpoints")
        
        if "data_size" in config and config["data_size"] <= 0:
            errors.append("Data size must be positive")
        
        return errors
    
    def _initialize_performance_metrics(self, nodes: List[HospitalNode]):
        """Initialize performance metrics for all nodes."""
        for node in nodes:
            self.performance_metrics[node.id] = {
                "requests_processed": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "current_load": 0.0,
                "last_health_check": time.time(),
                "circuit_breaker_state": "closed"
            }
    
    @asynccontextmanager
    async def request_session(self, user_id: str, request_data: Dict[str, Any]):
        """Context manager for request sessions with automatic cleanup."""
        session_id = str(uuid.uuid4())
        
        async with self.session_lock:
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "start_time": time.time(),
                "request_data": request_data,
                "status": "active"
            }
        
        try:
            yield session_id
        finally:
            async with self.session_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    session["end_time"] = time.time()
                    session["duration"] = session["end_time"] - session["start_time"]
                    session["status"] = "completed"
                    
                    # Archive session (remove from active)
                    del self.active_sessions[session_id]
    
    async def process_clinical_request_robust(
        self,
        user_id: str,
        clinical_prompt: str,
        department: str = "general",
        priority: int = 5,
        require_consensus: bool = False,
        max_privacy_budget: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Process clinical request with comprehensive robustness."""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Prepare request context
        request_context = {
            "user_id": user_id,
            "clinical_prompt": clinical_prompt,
            "department": department,
            "priority": priority,
            "require_consensus": require_consensus,
            "max_privacy_budget": max_privacy_budget,
            **kwargs
        }
        
        try:
            # Check system state
            if self.system_state not in [SystemState.HEALTHY, SystemState.DEGRADED]:
                return self._create_error_response(
                    request_id, "System unavailable", "SYSTEM_UNAVAILABLE"
                )
            
            # Rate limiting
            if self.rate_limiter:
                if not await self.rate_limiter.acquire(user_id):
                    await self._audit_log_event(
                        "rate_limit_exceeded", user_id, request_id, {"department": department}
                    )
                    return self._create_error_response(
                        request_id, "Rate limit exceeded", "RATE_LIMITED"
                    )
            
            # Input validation with circuit breaker
            if "validation" in self.circuit_breakers:
                validation_result = await self._execute_with_circuit_breaker(
                    "validation",
                    self._validate_request_with_retry,
                    user_id, clinical_prompt, department, max_privacy_budget, **kwargs
                )
            else:
                validation_result = self.validator.validate_clinical_request(
                    user_id, clinical_prompt, department, max_privacy_budget, **kwargs
                )
            
            if not validation_result.is_valid:
                await self._audit_log_event(
                    "validation_failed", user_id, request_id, {
                        "errors": validation_result.errors,
                        "security_issues": validation_result.security_issues
                    }
                )
                return self._create_error_response(
                    request_id, "Validation failed", "VALIDATION_ERROR",
                    details={"errors": validation_result.errors}
                )
            
            # Process request within session context
            async with self.request_session(user_id, request_context) as session_id:
                
                # Privacy budget check with circuit breaker
                if "privacy_accountant" in self.circuit_breakers:
                    budget_available = await self._execute_with_circuit_breaker(
                        "privacy_accountant",
                        self._check_privacy_budget,
                        user_id, max_privacy_budget, department
                    )
                else:
                    budget_available = self.budget_manager.can_query(department, max_privacy_budget)
                
                if not budget_available:
                    await self._audit_log_event(
                        "privacy_budget_exceeded", user_id, request_id, {"department": department}
                    )
                    return self._create_error_response(
                        request_id, "Privacy budget exceeded", "BUDGET_EXCEEDED"
                    )
                
                # Create inference request
                inference_request = InferenceRequest(
                    request_id=request_id,
                    user_id=user_id,
                    prompt=clinical_prompt,
                    model_name="medllama-7b",
                    max_privacy_budget=max_privacy_budget,
                    require_consensus=require_consensus,
                    priority=priority,
                    department=department
                )
                
                # Route request with circuit breaker and retry
                if "router" in self.circuit_breakers:
                    response = await self._execute_with_circuit_breaker(
                        "router",
                        self._route_request_with_retry,
                        inference_request
                    )
                else:
                    response = await self._route_request_with_retry(inference_request)
                
                # Update privacy budget
                self.budget_manager.deduct(department, response.privacy_cost)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                await self._update_performance_metrics(response.processing_nodes, processing_time, True)
                
                # Audit logging
                await self._audit_log_event(
                    "request_completed", user_id, request_id, {
                        "department": department,
                        "privacy_cost": response.privacy_cost,
                        "processing_nodes": response.processing_nodes,
                        "session_id": session_id
                    }
                )
                
                return {
                    "request_id": request_id,
                    "session_id": session_id,
                    "success": True,
                    "response": response.text,
                    "privacy_cost": response.privacy_cost,
                    "remaining_budget": response.remaining_budget,
                    "processing_nodes": response.processing_nodes,
                    "latency": response.latency,
                    "confidence_score": response.confidence_score,
                    "consensus_achieved": response.consensus_achieved,
                    "department": department,
                    "validation_score": validation_result.confidence_score,
                    "system_state": self.system_state.value
                }
        
        except Exception as e:
            error_context = self.error_handler.handle_error(
                e, "request_processing", ErrorSeverity.HIGH,
                user_id=user_id, request_id=request_id,
                additional_context=request_context
            )
            
            # Update failure metrics
            await self._update_performance_metrics([], 0, False)
            
            # Audit log error
            await self._audit_log_event(
                "request_failed", user_id, request_id, {
                    "error_id": error_context.error_id,
                    "error_type": error_context.error_type,
                    "department": department
                }
            )
            
            return self._create_error_response(
                request_id, f"Request processing failed: {error_context.message}", 
                "PROCESSING_ERROR", error_id=error_context.error_id
            )
    
    async def _validate_request_with_retry(self, *args, **kwargs) -> ValidationResult:
        """Validate request with retry logic."""
        for attempt in range(self.config.max_retry_attempts):
            try:
                return self.validator.validate_clinical_request(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retry_attempts - 1:
                    raise e
                
                delay = min(
                    self.config.retry_backoff_multiplier ** attempt,
                    self.config.retry_max_delay
                )
                await asyncio.sleep(delay)
    
    async def _check_privacy_budget(self, user_id: str, budget: float, department: str) -> bool:
        """Check privacy budget with error handling."""
        try:
            return self.budget_manager.can_query(department, budget)
        except Exception as e:
            self.error_handler.handle_error(
                e, "privacy_budget_check", ErrorSeverity.MEDIUM, user_id=user_id
            )
            # Fail-safe: deny request if budget check fails
            return False
    
    async def _route_request_with_retry(self, request: InferenceRequest) -> InferenceResponse:
        """Route request with retry and backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                return await self.router.route_request(request)
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on validation errors
                if "validation" in str(e).lower() or "budget" in str(e).lower():
                    raise e
                
                if attempt == self.config.max_retry_attempts - 1:
                    break
                
                delay = min(
                    self.config.retry_backoff_multiplier ** attempt,
                    self.config.retry_max_delay
                )
                await asyncio.sleep(delay)
        
        raise last_exception or RuntimeError("All retry attempts failed")
    
    async def _execute_with_circuit_breaker(self, breaker_name: str, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        circuit_breaker = self.circuit_breakers.get(breaker_name)
        if not circuit_breaker:
            return await func(*args, **kwargs)
        
        try:
            async with circuit_breaker:
                return await func(*args, **kwargs)
        except Exception as e:
            # Update circuit breaker state in metrics
            if breaker_name in self.performance_metrics:
                self.performance_metrics[breaker_name]["circuit_breaker_state"] = circuit_breaker.state.name
            raise e
    
    def _create_error_response(
        self, 
        request_id: str, 
        message: str, 
        error_code: str,
        details: Optional[Dict] = None,
        error_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "request_id": request_id,
            "success": False,
            "error": message,
            "error_code": error_code,
            "error_id": error_id,
            "details": details or {},
            "timestamp": time.time(),
            "system_state": self.system_state.value
        }
    
    async def _audit_log_event(
        self,
        event_type: str,
        user_id: str,
        request_id: str,
        additional_data: Optional[Dict] = None
    ):
        """Log audit event with proper retention and compliance."""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "request_id": request_id,
            "system_id": self.system_id,
            "additional_data": additional_data or {}
        }
        
        # Add compliance metadata
        if self.config.hipaa_compliance_mode:
            audit_entry["compliance"] = {
                "hipaa_covered": True,
                "retention_required": True,
                "access_logged": True
            }
        
        async with self.audit_lock:
            self.audit_log.append(audit_entry)
            
            # Rotate log if too large (basic implementation)
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]  # Keep recent half
    
    async def _update_performance_metrics(self, node_ids: List[str], latency: float, success: bool):
        """Update performance metrics with thread safety."""
        for node_id in node_ids:
            if node_id in self.performance_metrics:
                metrics = self.performance_metrics[node_id]
                
                # Update counters
                metrics["requests_processed"] += 1
                
                # Update average response time
                prev_avg = metrics["avg_response_time"]
                count = metrics["requests_processed"]
                metrics["avg_response_time"] = (prev_avg * (count - 1) + latency) / count
                
                # Update success rate
                if success:
                    prev_successes = metrics["success_rate"] * (count - 1)
                    metrics["success_rate"] = (prev_successes + 1) / count
                else:
                    prev_successes = metrics["success_rate"] * (count - 1)
                    metrics["success_rate"] = prev_successes / count
    
    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including robustness metrics."""
        error_stats = self.error_handler.get_error_statistics()
        
        circuit_breaker_states = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_breaker_states[name] = {
                "state": breaker.state.name,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time
            }
        
        return {
            "system_id": self.system_id,
            "system_state": self.system_state.value,
            "uptime": time.time() - self.start_time,
            "active_sessions": len(self.active_sessions),
            "configuration": {
                "security_validation": self.config.enable_security_validation,
                "circuit_breakers": self.config.circuit_breaker_enabled,
                "rate_limiting": self.config.rate_limit_enabled,
                "hipaa_compliance": self.config.hipaa_compliance_mode,
                "gdpr_compliance": self.config.gdpr_compliance_mode
            },
            "error_statistics": error_stats,
            "circuit_breaker_states": circuit_breaker_states,
            "performance_metrics": self.performance_metrics,
            "audit_log_size": len(self.audit_log),
            "timestamp": time.time()
        }
    
    async def graceful_shutdown(self, timeout: float = 60.0):
        """Perform graceful shutdown with proper cleanup."""
        self.logger.info("Initiating graceful shutdown...")
        self.system_state = SystemState.SHUTDOWN
        
        shutdown_start = time.time()
        
        # Wait for active sessions to complete
        while self.active_sessions and (time.time() - shutdown_start) < timeout:
            self.logger.info(f"Waiting for {len(self.active_sessions)} active sessions...")
            await asyncio.sleep(2)
        
        # Force cleanup remaining sessions
        if self.active_sessions:
            self.logger.warning(f"Force closing {len(self.active_sessions)} sessions")
            for session_id in list(self.active_sessions.keys()):
                await self._audit_log_event(
                    "session_force_closed", 
                    self.active_sessions[session_id]["user_id"],
                    session_id,
                    {"reason": "system_shutdown"}
                )
        
        # Stop health monitoring
        if self.health_checker:
            await self.health_checker.stop_monitoring()
        
        # Final audit log
        await self._audit_log_event(
            "system_shutdown", "system", self.system_id,
            {"uptime": time.time() - self.start_time}
        )
        
        self.logger.info("Graceful shutdown completed")


async def demo_robust_system():
    """Demonstrate robust system capabilities with error handling."""
    print("\n🛡️  Robust Enhanced System - Generation 2: MAKE IT ROBUST")
    print("=" * 70)
    
    # Configure robust system
    config = RobustSystemConfig(
        max_concurrent_requests=50,
        privacy_budget_per_user=10.0,
        enable_security_validation=True,
        enable_audit_logging=True,
        circuit_breaker_enabled=True,
        rate_limit_enabled=True,
        hipaa_compliance_mode=True,
        gdpr_compliance_mode=True
    )
    
    # Create robust system
    system = RobustFederatedSystem(config)
    
    # Hospital configuration with validation
    hospitals = [
        {
            "id": "hospital_robust_1",
            "endpoint": "https://robust1.federated.health:8443",
            "data_size": 75000,
            "compute_capacity": "6xA100",
            "department": "emergency",
            "region": "north_us"
        },
        {
            "id": "hospital_robust_2",
            "endpoint": "https://robust2.federated.health:8443",
            "data_size": 100000,
            "compute_capacity": "8xA100", 
            "department": "critical_care",
            "region": "south_us"
        }
    ]
    
    # Initialize with validation
    print("🏥 Initializing robust hospital network...")
    success, errors = await system.initialize_with_validation(hospitals)
    
    if not success:
        print(f"❌ Initialization failed:")
        for error in errors:
            print(f"   - {error}")
        return
    
    print("✅ Robust hospital network initialized successfully")
    
    # Demonstrate robust request processing
    print("\n🔒 Testing robust clinical request processing...")
    
    # Valid request
    valid_request = {
        "user_id": "dr_robust_001",
        "clinical_prompt": "62-year-old patient with acute MI, requires immediate intervention protocol",
        "department": "emergency",
        "priority": 9,
        "require_consensus": True,
        "max_privacy_budget": 0.15
    }
    
    result = await system.process_clinical_request_robust(**valid_request)
    if result["success"]:
        print(f"✅ Valid request processed successfully")
        print(f"   Request ID: {result['request_id'][:8]}")
        print(f"   Session ID: {result['session_id'][:8]}")
        print(f"   Validation score: {result['validation_score']:.2f}")
        print(f"   System state: {result['system_state']}")
    else:
        print(f"❌ Valid request failed: {result['error']}")
    
    # Test validation failures
    print("\n🚨 Testing validation and error handling...")
    
    invalid_requests = [
        {
            "name": "Empty prompt",
            "request": {
                "user_id": "dr_test",
                "clinical_prompt": "",  # Invalid: empty
                "department": "general",
                "max_privacy_budget": 0.1
            }
        },
        {
            "name": "Excessive privacy budget",
            "request": {
                "user_id": "dr_test",
                "clinical_prompt": "Test prompt",
                "department": "general",
                "max_privacy_budget": -0.1  # Invalid: negative
            }
        },
        {
            "name": "Suspicious content",
            "request": {
                "user_id": "dr_test",
                "clinical_prompt": "Patient data: DROP TABLE users; --",  # Security risk
                "department": "general",
                "max_privacy_budget": 0.1
            }
        }
    ]
    
    for test_case in invalid_requests:
        result = await system.process_clinical_request_robust(**test_case["request"])
        if not result["success"]:
            print(f"✅ {test_case['name']}: Correctly rejected - {result['error_code']}")
        else:
            print(f"❌ {test_case['name']}: Should have been rejected")
    
    # Demonstrate rate limiting
    print("\n⚡ Testing rate limiting...")
    rate_limit_requests = []
    for i in range(5):  # Quick succession of requests
        task = system.process_clinical_request_robust(
            user_id="dr_rate_test",
            clinical_prompt=f"Rate limit test request {i+1}",
            department="general",
            max_privacy_budget=0.05
        )
        rate_limit_requests.append(task)
    
    rate_results = await asyncio.gather(*rate_limit_requests)
    successful_rate = sum(1 for r in rate_results if r["success"])
    rate_limited = sum(1 for r in rate_results if r.get("error_code") == "RATE_LIMITED")
    
    print(f"✅ Rate limiting: {successful_rate} successful, {rate_limited} rate-limited")
    
    # Display comprehensive status
    print(f"\n📊 Comprehensive System Status:")
    status = system.get_comprehensive_system_status()
    print(f"   System state: {status['system_state']}")
    print(f"   Active sessions: {status['active_sessions']}")
    print(f"   Total errors: {status['error_statistics']['total_errors']}")
    print(f"   Error rate: {status['error_statistics']['error_rate']:.2f}/min")
    print(f"   Security validation: {status['configuration']['security_validation']}")
    print(f"   Circuit breakers: {status['configuration']['circuit_breakers']}")
    print(f"   HIPAA compliance: {status['configuration']['hipaa_compliance']}")
    print(f"   Audit log size: {status['audit_log_size']}")
    
    # Circuit breaker states
    if status['circuit_breaker_states']:
        print(f"   Circuit breaker states:")
        for name, state in status['circuit_breaker_states'].items():
            print(f"     - {name}: {state['state']}")
    
    # Graceful shutdown
    print(f"\n🛑 Testing graceful shutdown...")
    await system.graceful_shutdown(timeout=10.0)
    print("✅ Graceful shutdown completed successfully")
    
    print(f"\n🎉 Robust System Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_robust_system())