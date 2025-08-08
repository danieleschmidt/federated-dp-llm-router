"""
Comprehensive Error Handling and Recovery System

Implements robust error handling, automatic recovery, retry mechanisms,
and fault tolerance for the federated DP-LLM router.
"""

import asyncio
import functools
import time
import traceback
from typing import (
    Dict, List, Optional, Any, Callable, Union, Tuple, 
    Type, Set, Generic, TypeVar, Awaitable
)
from dataclasses import dataclass, field
from enum import Enum
import threading
import random
from datetime import datetime, timedelta


T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    NETWORK = "network"
    PRIVACY = "privacy"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    FEDERATION = "federation"
    QUANTUM = "quantum"
    DATABASE = "database"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE_SERVICE = "degrade_service"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    RESTART_COMPONENT = "restart_component"
    RESET_STATE = "reset_state"


@dataclass
class ErrorRule:
    """Rule for handling specific error types."""
    error_types: Set[Type[Exception]]
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_actions: List[RecoveryAction]
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    timeout_override: Optional[float] = None
    escalation_threshold: int = 5
    circuit_breaker_threshold: int = 10


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: str
    recovery_attempted: List[RecoveryAction] = field(default_factory=list)
    recovery_successful: bool = False
    resolution_time: Optional[float] = None
    escalated: bool = False


class RetryStrategy:
    """Configurable retry strategy."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[Set[Type[Exception]]] = None,
        stop_on: Optional[Set[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or set()
        self.stop_on = stop_on or set()
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry after this exception."""
        if attempt >= self.max_attempts:
            return False
        
        # Check stop conditions
        if self.stop_on and type(exception) in self.stop_on:
            return False
        
        # Check retry conditions
        if self.retry_on:
            return type(exception) in self.retry_on
        
        # Default: retry on most exceptions
        return True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if attempt <= 0:
            return 0
        
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset if half open
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e
    
    async def acall(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Async version of circuit breaker call."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset if half open
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time,
                "reset_timeout": self.reset_timeout
            }


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        self.error_rules: Dict[ErrorCategory, ErrorRule] = {}
        self.error_history: List[ErrorRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.escalation_handlers: Dict[ErrorSeverity, List[Callable]] = {}
        self.error_counter = 0
        self._lock = threading.Lock()
        
        # Setup default error rules
        self._setup_default_rules()
        
        # Error pattern detection
        self.error_patterns: Dict[str, List[float]] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[RecoveryAction, Callable] = {
            RecoveryAction.RETRY: self._retry_operation,
            RecoveryAction.FAILOVER: self._failover_operation,
            RecoveryAction.CIRCUIT_BREAK: self._circuit_break_operation,
            RecoveryAction.DEGRADE_SERVICE: self._degrade_service,
            RecoveryAction.ESCALATE: self._escalate_error,
            RecoveryAction.RESTART_COMPONENT: self._restart_component,
            RecoveryAction.RESET_STATE: self._reset_state
        }
    
    def _setup_default_rules(self):
        """Setup default error handling rules."""
        
        # Network errors
        self.error_rules[ErrorCategory.NETWORK] = ErrorRule(
            error_types={ConnectionError, TimeoutError, OSError},
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER],
            max_retries=3,
            retry_delay=2.0,
            circuit_breaker_threshold=5
        )
        
        # Privacy errors
        self.error_rules[ErrorCategory.PRIVACY] = ErrorRule(
            error_types={ValueError},  # Privacy budget exceeded, etc.
            category=ErrorCategory.PRIVACY,
            severity=ErrorSeverity.HIGH,
            recovery_actions=[RecoveryAction.ESCALATE],
            max_retries=0,  # No retries for privacy violations
            escalation_threshold=1
        )
        
        # Authentication errors
        self.error_rules[ErrorCategory.AUTHENTICATION] = ErrorRule(
            error_types={PermissionError},
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recovery_actions=[RecoveryAction.ESCALATE],
            max_retries=1,
            escalation_threshold=3
        )
        
        # Rate limit errors
        self.error_rules[ErrorCategory.RATE_LIMIT] = ErrorRule(
            error_types={},  # Custom exceptions
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.DEGRADE_SERVICE],
            max_retries=5,
            retry_delay=5.0,
            exponential_backoff=True
        )
        
        # Resource errors
        self.error_rules[ErrorCategory.RESOURCE] = ErrorRule(
            error_types={MemoryError, OSError},
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            recovery_actions=[RecoveryAction.RESTART_COMPONENT, RecoveryAction.ESCALATE],
            max_retries=1,
            escalation_threshold=2
        )
        
        # Timeout errors
        self.error_rules[ErrorCategory.TIMEOUT] = ErrorRule(
            error_types={asyncio.TimeoutError, TimeoutError},
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK],
            max_retries=2,
            retry_delay=1.0,
            circuit_breaker_threshold=3
        )
        
        # System errors
        self.error_rules[ErrorCategory.SYSTEM] = ErrorRule(
            error_types={SystemError, RuntimeError},
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            recovery_actions=[RecoveryAction.RESTART_COMPONENT, RecoveryAction.ESCALATE],
            max_retries=1,
            escalation_threshold=2
        )
    
    def register_error_rule(self, rule: ErrorRule):
        """Register custom error handling rule."""
        self.error_rules[rule.category] = rule
    
    def register_escalation_handler(
        self,
        severity: ErrorSeverity,
        handler: Callable[[ErrorRecord], None]
    ):
        """Register escalation handler for specific severity."""
        if severity not in self.escalation_handlers:
            self.escalation_handlers[severity] = []
        self.escalation_handlers[severity].append(handler)
    
    def categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize an exception."""
        exception_type = type(exception)
        
        for category, rule in self.error_rules.items():
            if exception_type in rule.error_types:
                return category
        
        # Check by exception attributes/message
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT
        elif any(keyword in error_message for keyword in ['connection', 'network', 'dns']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['permission', 'unauthorized', 'forbidden']):
            return ErrorCategory.AUTHENTICATION
        elif any(keyword in error_message for keyword in ['privacy', 'budget', 'epsilon']):
            return ErrorCategory.PRIVACY
        elif any(keyword in error_message for keyword in ['rate limit', 'throttle', 'quota']):
            return ErrorCategory.RATE_LIMIT
        elif any(keyword in error_message for keyword in ['memory', 'disk', 'space']):
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN
    
    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        operation: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        Handle an error with automatic recovery.
        
        Returns:
            Tuple of (recovery_successful, result_or_none)
        """
        category = self.categorize_error(exception)
        rule = self.error_rules.get(category)
        
        if not rule:
            # No rule found, create basic record and escalate
            error_record = self._create_error_record(
                exception, context, category, ErrorSeverity.MEDIUM
            )
            await self._escalate_error(error_record)
            return False, None
        
        # Create error record
        error_record = self._create_error_record(
            exception, context, category, rule.severity
        )
        
        # Store error record
        with self._lock:
            self.error_history.append(error_record)
            
            # Limit history size
            if len(self.error_history) > 10000:
                self.error_history = self.error_history[-10000:]
        
        # Track error patterns
        self._track_error_pattern(error_record)
        
        # Attempt recovery
        for action in rule.recovery_actions:
            try:
                success, result = await self._execute_recovery_action(
                    action, error_record, operation, args, kwargs
                )
                
                if success:
                    error_record.recovery_successful = True
                    error_record.resolution_time = time.time()
                    return True, result
                
                error_record.recovery_attempted.append(action)
                
            except Exception as recovery_error:
                # Recovery action itself failed
                await self._handle_recovery_failure(
                    recovery_error, action, error_record
                )
        
        # All recovery actions failed
        await self._handle_recovery_exhausted(error_record)
        return False, None
    
    def _create_error_record(
        self,
        exception: Exception,
        context: ErrorContext,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> ErrorRecord:
        """Create error record."""
        with self._lock:
            self.error_counter += 1
            error_id = f"error_{self.error_counter}_{int(time.time())}"
        
        return ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            category=category,
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc()
        )
    
    def _track_error_pattern(self, error_record: ErrorRecord):
        """Track error patterns for prediction."""
        pattern_key = f"{error_record.context.component}:{error_record.error_type}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = []
        
        self.error_patterns[pattern_key].append(error_record.timestamp)
        
        # Keep only recent errors (last 24 hours)
        cutoff_time = time.time() - (24 * 3600)
        self.error_patterns[pattern_key] = [
            t for t in self.error_patterns[pattern_key] if t > cutoff_time
        ]
    
    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Execute a recovery action."""
        recovery_func = self.recovery_strategies.get(action)
        if not recovery_func:
            return False, None
        
        try:
            return await recovery_func(error_record, operation, args, kwargs)
        except Exception:
            return False, None
    
    async def _retry_operation(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Retry the failed operation."""
        if not operation:
            return False, None
        
        category = error_record.category
        rule = self.error_rules.get(category)
        
        if not rule:
            return False, None
        
        retry_strategy = RetryStrategy(
            max_attempts=rule.max_retries,
            base_delay=rule.retry_delay,
            exponential_base=2.0 if rule.exponential_backoff else 1.0
        )
        
        for attempt in range(1, rule.max_retries + 1):
            try:
                # Wait before retry
                if attempt > 1:
                    delay = retry_strategy.get_delay(attempt - 1)
                    await asyncio.sleep(delay)
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                return True, result
                
            except Exception as retry_exception:
                if not retry_strategy.should_retry(retry_exception, attempt):
                    break
                
                # Continue to next attempt
                continue
        
        return False, None
    
    async def _failover_operation(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Attempt failover to backup system."""
        # Simulate failover logic
        # In practice, this would switch to backup nodes/services
        return False, None
    
    async def _circuit_break_operation(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Open circuit breaker for the component."""
        component_key = error_record.context.component
        
        if component_key not in self.circuit_breakers:
            self.circuit_breakers[component_key] = CircuitBreaker()
        
        # Circuit breaker is now active for this component
        return False, None
    
    async def _degrade_service(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Degrade service level to handle load."""
        # Simulate service degradation
        # In practice, this would reduce features/quality
        return False, None
    
    async def _escalate_error(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable] = None,
        args: tuple = (),
        kwargs: dict = None
    ) -> Tuple[bool, Any]:
        """Escalate error to appropriate handlers."""
        error_record.escalated = True
        
        handlers = self.escalation_handlers.get(error_record.severity, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_record)
                else:
                    handler(error_record)
            except Exception:
                # Handler failed, but continue with other handlers
                pass
        
        return False, None
    
    async def _restart_component(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Restart the affected component."""
        # Simulate component restart
        # In practice, this would restart services/containers
        await asyncio.sleep(2)  # Simulate restart time
        return False, None
    
    async def _reset_state(
        self,
        error_record: ErrorRecord,
        operation: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> Tuple[bool, Any]:
        """Reset component state."""
        # Simulate state reset
        return False, None
    
    async def _handle_recovery_failure(
        self,
        recovery_error: Exception,
        action: RecoveryAction,
        original_error: ErrorRecord
    ):
        """Handle failure of recovery action."""
        # Log recovery failure
        pass
    
    async def _handle_recovery_exhausted(self, error_record: ErrorRecord):
        """Handle case where all recovery actions failed."""
        # Final escalation
        await self._escalate_error(error_record)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        with self._lock:
            errors = self.error_history.copy()
        
        if not errors:
            return {"total_errors": 0}
        
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - (24 * 3600)
        
        recent_errors_1h = [e for e in errors if e.timestamp > hour_ago]
        recent_errors_24h = [e for e in errors if e.timestamp > day_ago]
        
        # Count by category
        category_counts = {}
        for error in errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Success rate
        successful_recoveries = len([e for e in errors if e.recovery_successful])
        recovery_rate = (successful_recoveries / len(errors)) * 100 if errors else 0
        
        return {
            "total_errors": len(errors),
            "errors_last_hour": len(recent_errors_1h),
            "errors_last_24h": len(recent_errors_24h),
            "error_rate_per_minute": len(recent_errors_1h) / 60.0,
            "recovery_success_rate": recovery_rate,
            "by_category": category_counts,
            "by_severity": severity_counts,
            "escalated_errors": len([e for e in errors if e.escalated]),
            "circuit_breakers": {
                name: breaker.get_state()
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def predict_failure_risk(self, component: str) -> float:
        """Predict failure risk for a component based on patterns."""
        patterns = [
            pattern for pattern_key, pattern in self.error_patterns.items()
            if pattern_key.startswith(f"{component}:")
        ]
        
        if not patterns:
            return 0.0
        
        # Simple prediction based on recent error frequency
        recent_cutoff = time.time() - 3600  # Last hour
        recent_errors = sum(
            len([t for t in pattern if t > recent_cutoff])
            for pattern in patterns
        )
        
        # Normalize to 0-1 scale
        return min(recent_errors / 10.0, 1.0)


def with_error_handling(
    category: Optional[ErrorCategory] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None
):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            context = ErrorContext(
                component=func.__module__,
                operation=func.__name__
            )
            
            try:
                if timeout:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    return await func(*args, **kwargs)
            except Exception as e:
                success, result = await error_handler.handle_error(
                    e, context, func, *args, **kwargs
                )
                if success:
                    return result
                else:
                    raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async wrapper
            async def async_version():
                return func(*args, **kwargs)
            
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler