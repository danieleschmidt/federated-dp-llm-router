"""
Resilience and Fault Tolerance

Advanced resilience patterns including circuit breakers, retry logic,
fallback mechanisms, and auto-healing for production deployment.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState

__all__ = [
    "CircuitBreaker", "CircuitBreakerState"
]