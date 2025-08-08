"""
Circuit Breaker Pattern

Implementation of circuit breaker pattern for preventing cascade failures
in federated LLM infrastructure with healthcare-grade reliability.
"""

import time
import asyncio
import logging
from typing import Callable, Any, Optional, Dict, Type
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from collections import deque
import threading

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass 
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout
    expected_exception: Type[Exception] = Exception
    

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for healthcare-critical services."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.recent_calls = deque(maxlen=100)  # Track recent call results
        self._lock = threading.RLock()
        
        # Metrics for monitoring
        self.total_calls = 0
        self.total_failures = 0
        self.total_timeouts = 0
        self.state_change_history = []
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._call_sync(func, *args, **kwargs)
            return sync_wrapper
    
    async def _call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if not self._allow_request():
                self._record_call(success=False, error_type="CircuitBreakerOpen")
                raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            self._record_success()
            self._record_call(success=True)
            return result
            
        except asyncio.TimeoutError as e:
            self.total_timeouts += 1
            self._record_failure()
            self._record_call(success=False, error_type="Timeout")
            logger.warning(f"Circuit breaker timeout for {func.__name__}: {e}")
            raise
            
        except self.config.expected_exception as e:
            self._record_failure()
            self._record_call(success=False, error_type=type(e).__name__)
            logger.warning(f"Circuit breaker failure for {func.__name__}: {e}")
            raise
            
        except Exception as e:
            # Unexpected exceptions don't count as failures unless configured
            self._record_call(success=False, error_type="UnexpectedException")
            logger.error(f"Unexpected error in circuit breaker for {func.__name__}: {e}")
            raise
    
    def _call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if not self._allow_request():
                self._record_call(success=False, error_type="CircuitBreakerOpen")
                raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            self._record_call(success=True)
            return result
            
        except self.config.expected_exception as e:
            self._record_failure()
            self._record_call(success=False, error_type=type(e).__name__)
            logger.warning(f"Circuit breaker failure for {func.__name__}: {e}")
            raise
            
        except Exception as e:
            self._record_call(success=False, error_type="UnexpectedException")
            logger.error(f"Unexpected error in circuit breaker for {func.__name__}: {e}")
            raise
    
    def _allow_request(self) -> bool:
        """Check if request should be allowed based on current state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            return self._should_attempt_reset()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True  # Allow limited requests in half-open state
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from OPEN to HALF_OPEN."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        if time_since_failure >= self.config.recovery_timeout:
            self._transition_to_half_open()
            return True
        return False
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open immediately goes back to open
                self._transition_to_open()
    
    def _record_call(self, success: bool, error_type: str = None):
        """Record call for metrics and monitoring."""
        call_record = {
            'timestamp': time.time(),
            'success': success,
            'error_type': error_type,
            'state': self.state.value
        }
        self.recent_calls.append(call_record)
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            logger.warning(f"Circuit breaker transitioning to OPEN state")
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self._record_state_change(CircuitBreakerState.OPEN)
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        if self.state != CircuitBreakerState.HALF_OPEN:
            logger.info(f"Circuit breaker transitioning to HALF_OPEN state")
            self.state = CircuitBreakerState.HALF_OPEN
            self.success_count = 0
            self.failure_count = 0
            self._record_state_change(CircuitBreakerState.HALF_OPEN)
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        if self.state != CircuitBreakerState.CLOSED:
            logger.info(f"Circuit breaker transitioning to CLOSED state")
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self._record_state_change(CircuitBreakerState.CLOSED)
    
    def _record_state_change(self, new_state: CircuitBreakerState):
        """Record state change for monitoring."""
        self.state_change_history.append({
            'timestamp': time.time(),
            'old_state': self.state.value if hasattr(self, 'state') else 'unknown',
            'new_state': new_state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        })
        
        # Keep only last 50 state changes
        if len(self.state_change_history) > 50:
            self.state_change_history.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        with self._lock:
            recent_success_rate = 0.0
            if self.recent_calls:
                recent_successes = sum(1 for call in self.recent_calls if call['success'])
                recent_success_rate = recent_successes / len(self.recent_calls)
            
            return {
                'state': self.state.value,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'total_timeouts': self.total_timeouts,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'recent_success_rate': recent_success_rate,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time else None,
                'state_changes': len(self.state_change_history),
                'recent_calls_count': len(self.recent_calls)
            }
    
    def force_open(self):
        """Force circuit breaker to OPEN state (for testing/emergency)."""
        with self._lock:
            logger.warning("Circuit breaker FORCED to OPEN state")
            self._transition_to_open()
    
    def force_close(self):
        """Force circuit breaker to CLOSED state (for testing/recovery)."""
        with self._lock:
            logger.info("Circuit breaker FORCED to CLOSED state")  
            self._transition_to_closed()
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.total_calls = 0
            self.total_failures = 0
            self.total_timeouts = 0
            self.recent_calls.clear()
            self.state_change_history.clear()
            logger.info("Circuit breaker reset to initial state")


class HealthcareCircuitBreakerManager:
    """Manager for multiple circuit breakers in healthcare system."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_configs = {
            'privacy_accountant': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2
            ),
            'quantum_planner': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3
            ),
            'federation_client': CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=120.0,
                success_threshold=5
            ),
            'model_inference': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                timeout=120.0,  # Longer timeout for model inference
                success_threshold=3
            )
        }
    
    def get_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.breakers:
            config = self.default_configs.get(service_name, CircuitBreakerConfig())
            self.breakers[service_name] = CircuitBreaker(config)
        return self.breakers[service_name]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        system_health = {
            'healthy_services': 0,
            'degraded_services': 0,
            'failed_services': 0,
            'services': {}
        }
        
        for name, breaker in self.breakers.items():
            metrics = breaker.get_metrics()
            
            if metrics['state'] == 'closed':
                system_health['healthy_services'] += 1
                service_status = 'healthy'
            elif metrics['state'] == 'half_open':
                system_health['degraded_services'] += 1
                service_status = 'degraded'
            else:  # open
                system_health['failed_services'] += 1
                service_status = 'failed'
            
            system_health['services'][name] = {
                'status': service_status,
                'metrics': metrics
            }
        
        return system_health