"""
Enhanced Error Handling System for Generation 2 Robustness

Implements comprehensive error classification, adaptive retry strategies,
graceful degradation, and self-healing mechanisms for production-ready
federated healthcare LLM infrastructure.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import traceback
import functools
import threading
from collections import defaultdict, deque
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    NETWORK = "network"
    PRIVACY = "privacy"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    EXTERNAL_SERVICE = "external_service"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION = "configuration"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"


@dataclass
class ErrorContext:
    """Context information for error analysis."""
    timestamp: float
    error_id: str
    component: str
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    node_id: Optional[str] = None
    privacy_budget_used: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorClassification:
    """Classification of an error."""
    category: ErrorCategory
    severity: ErrorSeverity
    retry_strategy: RetryStrategy
    is_recoverable: bool
    requires_escalation: bool
    suggested_action: str
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: float
    exception: Exception
    context: ErrorContext
    classification: ErrorClassification
    traceback_str: str
    resolution_attempted: bool = False
    resolution_successful: bool = False
    retry_count: int = 0


class ErrorClassifier:
    """Intelligent error classification system."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        self.pattern_weights = defaultdict(float)
        self.learning_enabled = True
        
    def _build_classification_rules(self) -> Dict[str, ErrorClassification]:
        """Build error classification rules for healthcare LLM systems."""
        return {
            # Network errors
            "connection_error": ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Retry with exponential backoff, check network connectivity",
                max_retries=5,
                backoff_base=2.0
            ),
            "timeout_error": ErrorClassification(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Increase timeout and retry, consider load balancing",
                max_retries=3,
                backoff_base=1.5
            ),
            
            # Privacy errors - critical for healthcare
            "privacy_budget_exceeded": ErrorClassification(
                category=ErrorCategory.PRIVACY,
                severity=ErrorSeverity.CRITICAL,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=True,
                suggested_action="Block operation, audit privacy usage, notify compliance team"
            ),
            "differential_privacy_violation": ErrorClassification(
                category=ErrorCategory.PRIVACY,
                severity=ErrorSeverity.CRITICAL,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=True,
                suggested_action="Immediate escalation to security team, audit trail"
            ),
            
            # Authentication/Authorization
            "authentication_failed": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=True,
                suggested_action="Security audit, potential credential compromise"
            ),
            "insufficient_permissions": ErrorClassification(
                category=ErrorCategory.AUTHORIZATION,
                severity=ErrorSeverity.MEDIUM,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=False,
                suggested_action="Verify user permissions, update access controls"
            ),
            
            # Resource errors
            "memory_exhausted": ErrorClassification(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                retry_strategy=RetryStrategy.ADAPTIVE,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Scale resources, optimize memory usage, load balancing",
                max_retries=2
            ),
            "gpu_out_of_memory": ErrorClassification(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                retry_strategy=RetryStrategy.ADAPTIVE,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Reduce batch size, distribute load, scale GPU resources",
                max_retries=2
            ),
            
            # Quantum-specific errors
            "quantum_decoherence": ErrorClassification(
                category=ErrorCategory.QUANTUM_DECOHERENCE,
                severity=ErrorSeverity.MEDIUM,
                retry_strategy=RetryStrategy.IMMEDIATE,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Re-initialize quantum state, fallback to classical routing",
                max_retries=3
            ),
            "entanglement_broken": ErrorClassification(
                category=ErrorCategory.QUANTUM_DECOHERENCE,
                severity=ErrorSeverity.LOW,
                retry_strategy=RetryStrategy.IMMEDIATE,
                is_recoverable=True,
                requires_escalation=False,
                suggested_action="Re-create entanglement, continue with independent tasks",
                max_retries=2
            ),
            
            # Validation errors
            "invalid_input": ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=False,
                suggested_action="Validate input format, provide user feedback"
            ),
            "hipaa_compliance_violation": ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.CRITICAL,
                retry_strategy=RetryStrategy.NO_RETRY,
                is_recoverable=False,
                requires_escalation=True,
                suggested_action="Immediate block, compliance audit, legal notification"
            )
        }
    
    def classify_error(self, exception: Exception, context: ErrorContext) -> ErrorClassification:
        """Classify an error based on exception and context."""
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        # Direct rule matching
        for pattern, classification in self.classification_rules.items():
            if pattern in error_str or pattern in error_type:
                if self.learning_enabled:
                    self._update_pattern_weights(pattern, True)
                return classification
        
        # Pattern-based classification
        return self._classify_by_patterns(exception, context)
    
    def _classify_by_patterns(self, exception: Exception, context: ErrorContext) -> ErrorClassification:
        """Classify error using pattern matching."""
        error_str = str(exception).lower()
        
        # Network-related patterns
        if any(keyword in error_str for keyword in ['connection', 'network', 'unreachable', 'dns']):
            return self.classification_rules["connection_error"]
        
        # Timeout patterns
        if any(keyword in error_str for keyword in ['timeout', 'timed out', 'deadline exceeded']):
            return self.classification_rules["timeout_error"]
        
        # Privacy patterns
        if any(keyword in error_str for keyword in ['privacy', 'budget', 'epsilon', 'differential']):
            return self.classification_rules["privacy_budget_exceeded"]
        
        # Resource patterns
        if any(keyword in error_str for keyword in ['memory', 'oom', 'resource', 'capacity']):
            return self.classification_rules["memory_exhausted"]
        
        # Authentication patterns
        if any(keyword in error_str for keyword in ['auth', 'credential', 'token', 'unauthorized']):
            return self.classification_rules["authentication_failed"]
        
        # Default classification for unknown errors
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            is_recoverable=True,
            requires_escalation=False,
            suggested_action="Monitor and analyze for pattern recognition",
            max_retries=3
        )
    
    def _update_pattern_weights(self, pattern: str, successful_match: bool):
        """Update pattern weights for machine learning."""
        if successful_match:
            self.pattern_weights[pattern] += 0.1
        else:
            self.pattern_weights[pattern] -= 0.05
        
        # Clamp weights
        self.pattern_weights[pattern] = max(0.0, min(1.0, self.pattern_weights[pattern]))


class AdaptiveRetryHandler:
    """Adaptive retry handler with multiple strategies."""
    
    def __init__(self):
        self.retry_history = defaultdict(list)
        self.success_rates = defaultdict(float)
        self.fibonacci_cache = [1, 1]
        
    async def execute_with_retry(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        classification: ErrorClassification = None,
        context: ErrorContext = None
    ) -> Any:
        """Execute function with adaptive retry logic."""
        if kwargs is None:
            kwargs = {}
        
        if classification is None or classification.retry_strategy == RetryStrategy.NO_RETRY:
            return await self._execute_once(func, args, kwargs)
        
        last_exception = None
        retry_count = 0
        
        while retry_count <= classification.max_retries:
            try:
                if retry_count > 0:
                    delay = self._calculate_delay(
                        retry_count, 
                        classification.retry_strategy,
                        classification.backoff_base,
                        classification.backoff_max,
                        context
                    )
                    await asyncio.sleep(delay)
                
                result = await self._execute_once(func, args, kwargs)
                
                # Record successful retry
                if retry_count > 0:
                    self._record_retry_success(context, retry_count)
                
                return result
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Record retry attempt
                self._record_retry_attempt(context, retry_count, e)
                
                if retry_count > classification.max_retries:
                    break
        
        # All retries exhausted
        raise last_exception
    
    async def _execute_once(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function once."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _calculate_delay(
        self,
        retry_count: int,
        strategy: RetryStrategy,
        base: float,
        max_delay: float,
        context: ErrorContext = None
    ) -> float:
        """Calculate retry delay based on strategy."""
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base * retry_count
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base * (2 ** (retry_count - 1))
        elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = base * self._get_fibonacci(retry_count)
        elif strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(retry_count, context)
        else:
            delay = base
        
        # Add jitter to prevent thundering herd
        if HAS_NUMPY:
            jitter = np.random.uniform(0.8, 1.2)
        else:
            import random
            jitter = random.uniform(0.8, 1.2)
        return min(delay * jitter, max_delay)
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self.fibonacci_cache) <= n:
            next_fib = self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
            self.fibonacci_cache.append(next_fib)
        return self.fibonacci_cache[n]
    
    def _calculate_adaptive_delay(self, retry_count: int, context: ErrorContext) -> float:
        """Calculate adaptive delay based on historical success rates."""
        if not context or not context.component:
            return 2.0 ** retry_count  # Default exponential
        
        component = context.component
        success_rate = self.success_rates.get(component, 0.5)
        
        # Higher delay for components with lower success rates
        base_delay = 1.0 / max(success_rate, 0.1)
        return base_delay * (1.5 ** retry_count)
    
    def _record_retry_attempt(self, context: ErrorContext, retry_count: int, exception: Exception):
        """Record retry attempt for learning."""
        if context and context.component:
            self.retry_history[context.component].append({
                'timestamp': time.time(),
                'retry_count': retry_count,
                'exception_type': type(exception).__name__,
                'successful': False
            })
    
    def _record_retry_success(self, context: ErrorContext, retry_count: int):
        """Record successful retry for learning."""
        if context and context.component:
            self.retry_history[context.component].append({
                'timestamp': time.time(),
                'retry_count': retry_count,
                'successful': True
            })
            
            # Update success rate
            component = context.component
            recent_attempts = self.retry_history[component][-50:]  # Last 50 attempts
            if recent_attempts:
                success_count = sum(1 for attempt in recent_attempts if attempt['successful'])
                self.success_rates[component] = success_count / len(recent_attempts)


class ErrorEscalationManager:
    """Manages error escalation and alerting."""
    
    def __init__(self):
        self.escalation_rules = self._build_escalation_rules()
        self.alert_callbacks = []
        self.escalated_errors = deque(maxlen=1000)
        
    def _build_escalation_rules(self) -> Dict[ErrorSeverity, Dict[str, Any]]:
        """Build escalation rules based on error severity."""
        return {
            ErrorSeverity.LOW: {
                'immediate_alert': False,
                'log_level': logging.INFO,
                'notification_delay': 300,  # 5 minutes
                'requires_acknowledgment': False
            },
            ErrorSeverity.MEDIUM: {
                'immediate_alert': False,
                'log_level': logging.WARNING,
                'notification_delay': 60,  # 1 minute
                'requires_acknowledgment': False
            },
            ErrorSeverity.HIGH: {
                'immediate_alert': True,
                'log_level': logging.ERROR,
                'notification_delay': 0,
                'requires_acknowledgment': True
            },
            ErrorSeverity.CRITICAL: {
                'immediate_alert': True,
                'log_level': logging.CRITICAL,
                'notification_delay': 0,
                'requires_acknowledgment': True,
                'page_on_call': True
            }
        }
    
    def escalate_error(self, error_record: ErrorRecord):
        """Escalate error according to severity and rules."""
        if not error_record.classification.requires_escalation:
            return
        
        severity = error_record.classification.severity
        rules = self.escalation_rules[severity]
        
        # Log the error
        logger = logging.getLogger(__name__)
        logger.log(rules['log_level'], 
                  f"Escalated error: {error_record.error_id} - {error_record.exception}")
        
        # Create escalation record
        escalation = {
            'error_record': error_record,
            'escalated_at': time.time(),
            'rules_applied': rules,
            'acknowledged': False
        }
        self.escalated_errors.append(escalation)
        
        # Send alerts
        if rules.get('immediate_alert', False):
            self._send_immediate_alert(error_record, rules)
        else:
            self._schedule_delayed_alert(error_record, rules)
    
    def _send_immediate_alert(self, error_record: ErrorRecord, rules: Dict[str, Any]):
        """Send immediate alert for critical errors."""
        alert_data = {
            'type': 'immediate',
            'error_id': error_record.error_id,
            'severity': error_record.classification.severity.value,
            'category': error_record.classification.category.value,
            'component': error_record.context.component,
            'message': str(error_record.exception),
            'timestamp': error_record.timestamp,
            'requires_acknowledgment': rules.get('requires_acknowledgment', False),
            'page_on_call': rules.get('page_on_call', False)
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
    
    def _schedule_delayed_alert(self, error_record: ErrorRecord, rules: Dict[str, Any]):
        """Schedule delayed alert for non-critical errors."""
        delay = rules.get('notification_delay', 60)
        
        async def delayed_alert():
            await asyncio.sleep(delay)
            
            alert_data = {
                'type': 'delayed',
                'error_id': error_record.error_id,
                'severity': error_record.classification.severity.value,
                'category': error_record.classification.category.value,
                'component': error_record.context.component,
                'message': str(error_record.exception),
                'timestamp': error_record.timestamp,
                'delay': delay
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    logging.error(f"Error in delayed alert callback: {e}")
        
        asyncio.create_task(delayed_alert())
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def acknowledge_error(self, error_id: str, acknowledger: str):
        """Acknowledge an escalated error."""
        for escalation in self.escalated_errors:
            if escalation['error_record'].error_id == error_id:
                escalation['acknowledged'] = True
                escalation['acknowledged_by'] = acknowledger
                escalation['acknowledged_at'] = time.time()
                break


class EnhancedErrorHandler:
    """Main enhanced error handling system."""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.retry_handler = AdaptiveRetryHandler()
        self.escalation_manager = ErrorEscalationManager()
        self.error_records = deque(maxlen=10000)
        self.error_counter = 0
        self._lock = threading.RLock()
        
        # Performance metrics
        self.total_errors = 0
        self.resolved_errors = 0
        self.escalated_errors = 0
        
    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        retry_func: Optional[Callable] = None,
        retry_args: tuple = (),
        retry_kwargs: dict = None
    ) -> Any:
        """Main error handling entry point."""
        if retry_kwargs is None:
            retry_kwargs = {}
        
        with self._lock:
            self.error_counter += 1
            error_id = f"ERR_{int(time.time())}_{self.error_counter}"
            self.total_errors += 1
        
        # Classify the error
        classification = self.classifier.classify_error(exception, context)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            exception=exception,
            context=context,
            classification=classification,
            traceback_str=traceback.format_exc()
        )
        
        self.error_records.append(error_record)
        
        # Log the error
        logger = logging.getLogger(__name__)
        logger.error(f"Error {error_id}: {exception} in {context.component}")
        
        # Handle escalation
        if classification.requires_escalation:
            self.escalation_manager.escalate_error(error_record)
            self.escalated_errors += 1
        
        # Attempt recovery if retry function provided
        if retry_func and classification.is_recoverable:
            try:
                result = await self.retry_handler.execute_with_retry(
                    retry_func, retry_args, retry_kwargs, classification, context
                )
                error_record.resolution_attempted = True
                error_record.resolution_successful = True
                self.resolved_errors += 1
                return result
            except Exception as retry_exception:
                error_record.resolution_attempted = True
                error_record.resolution_successful = False
                # Log retry failure
                logger.error(f"Retry failed for {error_id}: {retry_exception}")
                raise retry_exception
        
        # Re-raise original exception if no recovery attempted
        raise exception
    
    def create_decorator(self, component: str, operation: str):
        """Create decorator for automatic error handling."""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = ErrorContext(
                    timestamp=time.time(),
                    error_id="",  # Will be set in handle_error
                    component=component,
                    operation=operation
                )
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    return await self.handle_error(e, context, func, args, kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                context = ErrorContext(
                    timestamp=time.time(),
                    error_id="",
                    component=component,
                    operation=operation
                )
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Convert to async for handling
                    async def async_handle():
                        return await self.handle_error(e, context, func, args, kwargs)
                    
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(async_handle())
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            recent_errors = [e for e in self.error_records 
                           if time.time() - e.timestamp <= 3600]  # Last hour
            
            category_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            component_errors = defaultdict(int)
            
            for error in recent_errors:
                category_counts[error.classification.category.value] += 1
                severity_counts[error.classification.severity.value] += 1
                component_errors[error.context.component] += 1
            
            resolution_rate = self.resolved_errors / max(self.total_errors, 1)
            escalation_rate = self.escalated_errors / max(self.total_errors, 1)
            
            return {
                'total_errors': self.total_errors,
                'resolved_errors': self.resolved_errors,
                'escalated_errors': self.escalated_errors,
                'resolution_rate': resolution_rate,
                'escalation_rate': escalation_rate,
                'recent_errors_count': len(recent_errors),
                'error_categories': dict(category_counts),
                'error_severities': dict(severity_counts),
                'component_errors': dict(component_errors),
                'timestamp': time.time()
            }


# Global enhanced error handler instance
enhanced_error_handler = EnhancedErrorHandler()