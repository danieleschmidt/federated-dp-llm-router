"""Performance optimization components."""

from .caching import CacheManager
from .connection_pool import ConnectionPoolManager
from .performance_optimizer import PerformanceOptimizer

# Conditional imports
try:
    from .caching import RedisCache
    __all__ = ["CacheManager", "RedisCache", "ConnectionPoolManager", "PerformanceOptimizer"]
except ImportError:
    __all__ = ["CacheManager", "ConnectionPoolManager", "PerformanceOptimizer"]