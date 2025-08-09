"""
Comprehensive Error Handling System

Provides robust error handling, retry mechanisms, circuit breakers,
and graceful degradation for the federated system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import random
import traceback
from contextlib import asynccontextmanager

from ..monitoring.logging_config import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PRIVACY_BUDGET = "privacy_budget"
    MODEL_INFERENCE = "model_inference"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    node_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    context: ErrorContext
    timestamp: float
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False


class FederatedError(Exception):
    """Base exception for federated system."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.details = details or {}
        self.timestamp = time.time()


class NetworkError(FederatedError):
    """Network-related errors."""
    
    def __init__(self, message: str, node_id: str = None, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, **kwargs)
        if node_id:
            self.context.node_id = node_id


class PrivacyBudgetError(FederatedError):
    """Privacy budget related errors."""
    
    def __init__(self, message: str, user_id: str = None, **kwargs):
        super().__init__(message, ErrorCategory.PRIVACY_BUDGET, 
                        ErrorSeverity.HIGH, **kwargs)
        if user_id:
            self.context.user_id = user_id


class ModelInferenceError(FederatedError):
    """Model inference related errors."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_INFERENCE, **kwargs)
        if model_name:
            self.details["model_name"] = model_name


class CircuitBreakerOpenError(FederatedError):
    """Circuit breaker is open."""
    
    def __init__(self, service_name: str):
        super().__init__(
            f"Circuit breaker open for {service_name}",
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorSeverity.HIGH
        )
        self.details["service_name"] = service_name


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 3  # Consecutive successes needed to close


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.logger = get_logger(f"circuit_breaker.{name}")
    
    def __call__(self, func):
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.logger.info(f"Circuit breaker {self.name} half-open")
            else:
                raise CircuitBreakerOpenError(self.name)
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} closed")
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker {self.name} opened after {self.failure_count} failures"
            )


@dataclass 
class RetryConfig:
    """Retry mechanism configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_exceptions: tuple = (NetworkError, Exception)


class RetryHandler:
    """Exponential backoff retry handler."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger("retry_handler")
    
    def __call__(self, func):
        """Decorator for retry mechanism."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    self.logger.error(f"Operation failed after {self.config.max_attempts} attempts")
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Operation failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** attempt)
        else:
            delay = self.config.base_delay
        
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter up to 25% of delay
            jitter = delay * 0.25 * random.random()
            delay += jitter
        
        return delay


class ErrorAggregator:
    """Aggregates and analyzes error patterns."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.errors: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = get_logger("error_aggregator")
    
    def add_error(self, error: ErrorInfo):
        """Add error to aggregation."""
        self.errors.append(error)
        
        # Maintain window size
        if len(self.errors) > self.window_size:
            old_error = self.errors.pop(0)
            key = f"{old_error.category.value}:{old_error.severity.value}"
            self.error_counts[key] = max(0, self.error_counts.get(key, 0) - 1)
        
        # Update counts
        key = f"{error.category.value}:{error.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Log critical errors immediately
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Critical error detected: {error.message}",
                extra={
                    "error_id": error.error_id,
                    "category": error.category.value,
                    "context": error.context.__dict__,
                    "details": error.details
                }
            )
    
    def get_error_rate(self, category: ErrorCategory = None, 
                      time_window: int = 300) -> float:
        """Get error rate for category in time window."""
        current_time = time.time()
        recent_errors = [
            e for e in self.errors
            if current_time - e.timestamp <= time_window
        ]
        
        if category:
            recent_errors = [e for e in recent_errors if e.category == category]
        
        if not recent_errors:
            return 0.0
        
        # Assume we have some total operation count to calculate rate
        # For now, return error count per minute
        return len(recent_errors) / (time_window / 60.0)
    
    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent error patterns."""
        error_patterns = {}
        
        for error in self.errors[-100:]:  # Last 100 errors
            pattern_key = f"{error.category.value}:{error.message[:100]}"
            if pattern_key in error_patterns:
                error_patterns[pattern_key]["count"] += 1
                error_patterns[pattern_key]["last_seen"] = error.timestamp
            else:
                error_patterns[pattern_key] = {
                    "category": error.category.value,
                    "message": error.message,
                    "count": 1,
                    "first_seen": error.timestamp,
                    "last_seen": error.timestamp,
                    "severity": error.severity.value
                }
        
        # Sort by count and return top patterns
        sorted_patterns = sorted(
            error_patterns.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return sorted_patterns[:limit]


class ErrorHandler:
    """Central error handling system."""
    
    def __init__(self):
        self.logger = get_logger("error_handler")
        self.aggregator = ErrorAggregator()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_callbacks: List[Callable] = []
    
    def add_error_callback(self, callback: Callable[[ErrorInfo], None]):
        """Add callback for error events."""
        self.error_callbacks.append(callback)
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config, name)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    async def handle_error(self, error: Exception, context: Optional[ErrorContext] = None,
                          category: Optional[ErrorCategory] = None,
                          severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
        """Handle and process error."""
        
        # Extract information from FederatedError
        if isinstance(error, FederatedError):
            error_info = ErrorInfo(
                error_id=f"err_{int(time.time())}_{hash(str(error)) % 10000}",
                category=error.category,
                severity=error.severity,
                message=error.message,
                details=error.details,
                context=error.context,
                timestamp=error.timestamp,
                stack_trace=traceback.format_exc()
            )
        else:
            # Handle generic exceptions
            error_info = ErrorInfo(
                error_id=f"err_{int(time.time())}_{hash(str(error)) % 10000}",
                category=category or ErrorCategory.SYSTEM,
                severity=severity or ErrorSeverity.MEDIUM,
                message=str(error),
                details={"exception_type": type(error).__name__},
                context=context or ErrorContext(),
                timestamp=time.time(),
                stack_trace=traceback.format_exc()
            )
        
        # Add to aggregator
        self.aggregator.add_error(error_info)
        
        # Log error
        log_level = logging.CRITICAL if error_info.severity == ErrorSeverity.CRITICAL else logging.ERROR
        self.logger.log(
            log_level,
            f"Error handled: {error_info.message}",
            extra={
                "error_id": error_info.error_id,
                "category": error_info.category.value,
                "severity": error_info.severity.value,
                "context": error_info.context.__dict__,
                "details": error_info.details,
                "stack_trace": error_info.stack_trace
            }
        )
        
        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
        
        return error_info
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors": len(self.aggregator.errors),
            "error_counts": self.aggregator.error_counts.copy(),
            "error_rate_5min": self.aggregator.get_error_rate(time_window=300),
            "top_errors": self.aggregator.get_top_errors(),
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            }
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


@asynccontextmanager
async def error_context(**context_kwargs):
    """Context manager for error handling with context."""
    context = ErrorContext(**context_kwargs)
    error_handler = get_error_handler()
    
    try:
        yield context
    except Exception as e:
        await error_handler.handle_error(e, context)
        raise


def handle_errors(category: ErrorCategory = None, 
                 severity: ErrorSeverity = None,
                 reraise: bool = True):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                await error_handler.handle_error(e, category=category, severity=severity)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


def create_retry_handler(max_attempts: int = 3, base_delay: float = 1.0) -> RetryHandler:
    """Create a retry handler with common configuration."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=(NetworkError, ModelInferenceError, Exception)
    )
    return RetryHandler(config)


def create_circuit_breaker(name: str, failure_threshold: int = 5) -> CircuitBreaker:
    """Create and register a circuit breaker."""
    config = CircuitBreakerConfig(failure_threshold=failure_threshold)
    error_handler = get_error_handler()
    return error_handler.register_circuit_breaker(name, config)