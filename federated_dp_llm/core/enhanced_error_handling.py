"""
Enhanced Error Handling and Resilience for Federated DP-LLM System

Implements comprehensive error handling, circuit breakers, retry mechanisms,
and graceful degradation for production robustness.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
import traceback

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for handling strategies."""
    PRIVACY_BUDGET_EXHAUSTED = "privacy_budget_exhausted"
    AUTHENTICATION_FAILED = "authentication_failed"
    NETWORK_TIMEOUT = "network_timeout"
    MODEL_OVERLOAD = "model_overload"
    VALIDATION_ERROR = "validation_error"
    QUANTUM_COHERENCE_LOST = "quantum_coherence_lost"
    NODE_UNAVAILABLE = "node_unavailable"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INTERNAL_ERROR = "internal_error"


class ErrorSeverity(Enum):
    """Error severity levels for response prioritization."""
    CRITICAL = "critical"      # System-wide impact
    HIGH = "high"             # Service degradation
    MEDIUM = "medium"         # Recoverable issues
    LOW = "low"               # Minor issues
    INFO = "info"             # Informational


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: ErrorType
    severity: ErrorSeverity
    timestamp: float
    user_id: Optional[str]
    request_id: Optional[str]
    component: str
    details: Dict[str, Any]
    retry_count: int = 0
    recoverable: bool = True


class FederatedError(Exception):
    """Base exception for federated system errors."""
    
    def __init__(self, message: str, error_type: ErrorType, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = time.time()


class PrivacyBudgetExhaustedException(FederatedError):
    """Exception for privacy budget exhaustion."""
    
    def __init__(self, user_id: str, requested: float, available: float):
        super().__init__(
            f"Privacy budget exhausted for user {user_id}. "
            f"Requested: {requested:.3f}, Available: {available:.3f}",
            ErrorType.PRIVACY_BUDGET_EXHAUSTED,
            ErrorSeverity.HIGH,
            {"user_id": user_id, "requested": requested, "available": available},
            recoverable=False
        )


class QuantumCoherenceLostException(FederatedError):
    """Exception for quantum coherence loss in planning system."""
    
    def __init__(self, task_id: str, coherence_level: float):
        super().__init__(
            f"Quantum coherence lost for task {task_id}. "
            f"Coherence level: {coherence_level:.3f}",
            ErrorType.QUANTUM_COHERENCE_LOST,
            ErrorSeverity.MEDIUM,
            {"task_id": task_id, "coherence_level": coherence_level},
            recoverable=True
        )


class NodeUnavailableException(FederatedError):
    """Exception for unavailable federated nodes."""
    
    def __init__(self, node_id: str, last_seen: float):
        super().__init__(
            f"Node {node_id} unavailable. Last seen: {time.time() - last_seen:.1f}s ago",
            ErrorType.NODE_UNAVAILABLE,
            ErrorSeverity.HIGH,
            {"node_id": node_id, "last_seen": last_seen},
            recoverable=True
        )


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker for service protection."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise FederatedError(
                    f"Circuit breaker {self.name} is OPEN",
                    ErrorType.RESOURCE_EXHAUSTED,
                    ErrorSeverity.HIGH
                )
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
        
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to failures")


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except FederatedError as e:
                last_exception = e
                
                # Don't retry non-recoverable errors
                if not e.recoverable:
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if self.config.jitter:
                    import random
                    delay *= (0.5 + 0.5 * random.random())
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Wrap unexpected exceptions
                wrapped_error = FederatedError(
                    f"Unexpected error: {str(e)}",
                    ErrorType.INTERNAL_ERROR,
                    ErrorSeverity.HIGH,
                    {"original_error": str(e)},
                    recoverable=True
                )
                last_exception = wrapped_error
                
                if attempt == self.config.max_attempts - 1:
                    raise wrapped_error
                
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                await asyncio.sleep(delay)
        
        raise last_exception


class GracefulDegradation:
    """Handles graceful service degradation."""
    
    def __init__(self):
        self.degradation_levels = {
            "full": {"privacy_epsilon": 0.1, "consensus_required": True, "quantum_enabled": True},
            "high": {"privacy_epsilon": 0.2, "consensus_required": True, "quantum_enabled": False},
            "medium": {"privacy_epsilon": 0.5, "consensus_required": False, "quantum_enabled": False},
            "low": {"privacy_epsilon": 1.0, "consensus_required": False, "quantum_enabled": False},
            "minimal": {"privacy_epsilon": 2.0, "consensus_required": False, "quantum_enabled": False}
        }
        self.current_level = "full"
    
    def degrade_service(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Determine appropriate service degradation level."""
        if error_context.error_type == ErrorType.PRIVACY_BUDGET_EXHAUSTED:
            self.current_level = "medium"
        elif error_context.error_type == ErrorType.QUANTUM_COHERENCE_LOST:
            self.current_level = "high"
        elif error_context.error_type == ErrorType.NODE_UNAVAILABLE:
            self.current_level = "low"
        elif error_context.severity == ErrorSeverity.CRITICAL:
            self.current_level = "minimal"
        
        logger.info(f"Service degraded to level: {self.current_level}")
        return self.degradation_levels[self.current_level]
    
    def can_recover(self) -> bool:
        """Check if service can recover to higher level."""
        return self.current_level != "full"
    
    def attempt_recovery(self):
        """Attempt to recover to higher service level."""
        levels = ["minimal", "low", "medium", "high", "full"]
        current_index = levels.index(self.current_level)
        
        if current_index < len(levels) - 1:
            self.current_level = levels[current_index + 1]
            logger.info(f"Service recovery attempted to level: {self.current_level}")


class ErrorAnalyzer:
    """Analyzes error patterns for system optimization."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, int] = {}
        self.analysis_window = 3600  # 1 hour
    
    def record_error(self, error_context: ErrorContext):
        """Record error for pattern analysis."""
        self.error_history.append(error_context)
        
        # Clean old errors
        cutoff_time = time.time() - self.analysis_window
        self.error_history = [
            ctx for ctx in self.error_history 
            if ctx.timestamp > cutoff_time
        ]
        
        # Update patterns
        pattern_key = f"{error_context.error_type.value}_{error_context.component}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
    
    def get_error_insights(self) -> Dict[str, Any]:
        """Get insights from error analysis."""
        if not self.error_history:
            return {"total_errors": 0, "patterns": {}, "recommendations": []}
        
        total_errors = len(self.error_history)
        error_types = {}
        severity_distribution = {}
        
        for ctx in self.error_history:
            error_types[ctx.error_type.value] = error_types.get(ctx.error_type.value, 0) + 1
            severity_distribution[ctx.severity.value] = severity_distribution.get(ctx.severity.value, 0) + 1
        
        # Generate recommendations
        recommendations = []
        if error_types.get("node_unavailable", 0) > 5:
            recommendations.append("Consider adding more federated nodes for redundancy")
        if error_types.get("privacy_budget_exhausted", 0) > 10:
            recommendations.append("Review privacy budget allocation strategy")
        if error_types.get("quantum_coherence_lost", 0) > 3:
            recommendations.append("Optimize quantum task planning parameters")
        
        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "severity_distribution": severity_distribution,
            "patterns": dict(self.error_patterns),
            "recommendations": recommendations
        }


class EnhancedErrorHandler:
    """Main error handling orchestrator."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler(RetryConfig())
        self.degradation = GracefulDegradation()
        self.analyzer = ErrorAnalyzer()
        
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker for a service."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
    
    @asynccontextmanager
    async def handle_errors(self, component: str, user_id: str = None, request_id: str = None):
        """Context manager for comprehensive error handling."""
        try:
            yield
        except FederatedError as e:
            error_context = ErrorContext(
                error_type=e.error_type,
                severity=e.severity,
                timestamp=e.timestamp,
                user_id=user_id,
                request_id=request_id,
                component=component,
                details=e.context,
                recoverable=e.recoverable
            )
            
            await self._handle_federated_error(error_context)
            raise
        
        except Exception as e:
            # Handle unexpected errors
            error_context = ErrorContext(
                error_type=ErrorType.INTERNAL_ERROR,
                severity=ErrorSeverity.HIGH,
                timestamp=time.time(),
                user_id=user_id,
                request_id=request_id,
                component=component,
                details={"original_error": str(e), "traceback": traceback.format_exc()},
                recoverable=True
            )
            
            await self._handle_federated_error(error_context)
            
            # Wrap and re-raise
            wrapped_error = FederatedError(
                f"Internal error in {component}: {str(e)}",
                ErrorType.INTERNAL_ERROR,
                ErrorSeverity.HIGH,
                error_context.details
            )
            raise wrapped_error
    
    async def _handle_federated_error(self, error_context: ErrorContext):
        """Handle federated system error."""
        # Record error for analysis
        self.analyzer.record_error(error_context)
        
        # Log error appropriately
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.INFO: logging.INFO
        }.get(error_context.severity, logging.WARNING)
        
        logger.log(log_level, 
                  f"Error in {error_context.component}: {error_context.error_type.value} "
                  f"(Severity: {error_context.severity.value})")
        
        # Apply degradation if necessary
        if error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            degradation_config = self.degradation.degrade_service(error_context)
            logger.info(f"Applied service degradation: {degradation_config}")
    
    async def execute_with_protection(self, func: Callable, component: str, 
                                    circuit_breaker: str = None, 
                                    retry: bool = True,
                                    user_id: str = None,
                                    request_id: str = None,
                                    *args, **kwargs):
        """Execute function with full error protection."""
        async with self.handle_errors(component, user_id, request_id):
            
            # Apply circuit breaker if specified
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                if retry:
                    return await self.retry_handler.retry(
                        self.circuit_breakers[circuit_breaker].call,
                        func, *args, **kwargs
                    )
                else:
                    return await self.circuit_breakers[circuit_breaker].call(func, *args, **kwargs)
            
            # Apply retry if requested
            elif retry:
                return await self.retry_handler.retry(func, *args, **kwargs)
            
            # Direct execution
            else:
                return await func(*args, **kwargs)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        circuit_breaker_status = {
            name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        degradation_status = {
            "current_level": self.degradation.current_level,
            "can_recover": self.degradation.can_recover()
        }
        
        error_insights = self.analyzer.get_error_insights()
        
        return {
            "circuit_breakers": circuit_breaker_status,
            "service_degradation": degradation_status,
            "error_analysis": error_insights,
            "timestamp": time.time()
        }


# Global error handler instance
global_error_handler = EnhancedErrorHandler()


def get_error_handler() -> EnhancedErrorHandler:
    """Get the global error handler instance."""
    return global_error_handler