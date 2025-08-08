"""
Advanced Caching System for Federated DP-LLM Router

Implements multi-layer caching with privacy-aware cache management,
distributed caching, and intelligent cache warming for optimal performance.
"""

import asyncio
import hashlib
import json
import time
import pickle
import zlib
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import OrderedDict, defaultdict
import math


class CacheType(Enum):
    """Types of cache storage."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    PRIVACY_TTL = "privacy_ttl"  # Privacy-aware TTL
    ADAPTIVE = "adaptive"  # Machine learning based


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    privacy_level: str = "public"
    user_id: Optional[str] = None
    department: Optional[str] = None
    compressed: bool = False
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class PrivacyAwareCaching:
    """Privacy-aware caching logic."""
    
    def __init__(self):
        # Privacy levels and their TTL multipliers
        self.privacy_ttl_multipliers = {
            "public": 1.0,
            "internal": 0.5,
            "confidential": 0.1,
            "restricted": 0.05,
            "private": 0.0  # No caching
        }
        
        # Department-specific cache policies
        self.department_policies = {
            "emergency": {"max_ttl": 300, "privacy_multiplier": 0.2},  # 5 min max
            "research": {"max_ttl": 3600, "privacy_multiplier": 0.8},  # 1 hour max
            "general": {"max_ttl": 1800, "privacy_multiplier": 0.5},   # 30 min max
        }
    
    def should_cache(self, privacy_level: str, user_id: str, department: str) -> bool:
        """Determine if content should be cached."""
        if privacy_level == "private":
            return False
        
        # Check department policies
        dept_policy = self.department_policies.get(department, {})
        if dept_policy.get("no_cache", False):
            return False
        
        return True
    
    def calculate_ttl(
        self,
        base_ttl: float,
        privacy_level: str,
        department: str,
        data_sensitivity: float = 0.5
    ) -> float:
        """Calculate privacy-aware TTL."""
        if privacy_level == "private":
            return 0
        
        # Apply privacy multiplier
        privacy_multiplier = self.privacy_ttl_multipliers.get(privacy_level, 0.1)
        
        # Apply department policy
        dept_policy = self.department_policies.get(department, {})
        dept_multiplier = dept_policy.get("privacy_multiplier", 1.0)
        max_ttl = dept_policy.get("max_ttl", float('inf'))
        
        # Calculate final TTL
        calculated_ttl = base_ttl * privacy_multiplier * dept_multiplier * (1 - data_sensitivity)
        
        return min(calculated_ttl, max_ttl)


class IntelligentCache:
    """High-performance intelligent cache with privacy awareness."""
    
    def __init__(
        self,
        max_size_mb: int = 512,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        enable_compression: bool = True,
        compression_threshold: int = 1024,
        enable_analytics: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_analytics = enable_analytics
        
        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_counts = defaultdict(int)  # For LFU
        self.size_tracker = 0
        
        # Privacy and security
        self.privacy_manager = PrivacyAwareCaching()
        
        # Performance tracking
        self.stats = CacheStats()
        self.performance_history = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache warming
        self.warm_cache_enabled = True
        self.warming_tasks = set()
        
        # Analytics
        if enable_analytics:
            self.access_patterns = defaultdict(list)
            self.hit_patterns = defaultdict(int)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if beneficial."""
        if len(data) < self.compression_threshold:
            return data
        
        compressed = zlib.compress(data)
        if len(compressed) < len(data) * 0.9:  # At least 10% reduction
            return compressed
        
        return data
    
    def _decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """Decompress data if it was compressed."""
        if compressed:
            return zlib.decompress(data)
        return data
    
    def _serialize_value(self, value: Any) -> Tuple[bytes, bool]:
        """Serialize and optionally compress value."""
        serialized = pickle.dumps(value)
        
        if self.enable_compression:
            compressed = self._compress_data(serialized)
            was_compressed = len(compressed) < len(serialized)
            return compressed, was_compressed
        
        return serialized, False
    
    def _deserialize_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress and deserialize value."""
        decompressed = self._decompress_data(data, compressed)
        return pickle.loads(decompressed)
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return self.size_tracker > self.max_size_bytes
    
    def _select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select entries for eviction based on policy."""
        if not self.entries:
            return []
        
        if self.eviction_policy == EvictionPolicy.LRU:
            return list(self.access_order.keys())[:count]
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            sorted_entries = sorted(
                self.entries.keys(),
                key=lambda k: self.access_counts[k]
            )
            return sorted_entries[:count]
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            expired = [
                key for key, entry in self.entries.items()
                if entry.is_expired
            ]
            if len(expired) >= count:
                return expired[:count]
            
            # If not enough expired, fall back to LRU
            return list(self.access_order.keys())[:count]
        
        elif self.eviction_policy == EvictionPolicy.PRIVACY_TTL:
            # Prioritize evicting private/sensitive data
            privacy_priority = {
                "private": 0, "restricted": 1, "confidential": 2,
                "internal": 3, "public": 4
            }
            
            sorted_entries = sorted(
                self.entries.keys(),
                key=lambda k: (
                    privacy_priority.get(self.entries[k].privacy_level, 5),
                    self.entries[k].age_seconds
                )
            )
            return sorted_entries[:count]
        
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            return self._adaptive_eviction_selection(count)
        
        else:
            return list(self.access_order.keys())[:count]
    
    def _adaptive_eviction_selection(self, count: int) -> List[str]:
        """Intelligent eviction selection using multiple factors."""
        candidates = []
        
        for key, entry in self.entries.items():
            # Calculate eviction score (lower = more likely to evict)
            score = 0.0
            
            # Factor 1: Age (older = higher eviction chance)
            age_factor = entry.age_seconds / 3600  # Normalize to hours
            score += age_factor * 0.3
            
            # Factor 2: Access frequency (less frequent = higher eviction chance)
            freq_factor = 1.0 / (entry.access_count + 1)
            score += freq_factor * 0.3
            
            # Factor 3: Privacy level (more private = higher eviction chance)
            privacy_factor = {
                "private": 1.0, "restricted": 0.8, "confidential": 0.6,
                "internal": 0.4, "public": 0.2
            }.get(entry.privacy_level, 0.5)
            score += privacy_factor * 0.2
            
            # Factor 4: Size (larger = higher eviction chance)
            size_factor = entry.size_bytes / (1024 * 1024)  # Normalize to MB
            score += size_factor * 0.1
            
            # Factor 5: Recency (recently accessed = lower eviction chance)
            recency_factor = (time.time() - entry.last_accessed) / 3600
            score += recency_factor * 0.1
            
            candidates.append((key, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in candidates[:count]]
    
    def _evict_entries(self, keys: List[str]):
        """Evict specified cache entries."""
        for key in keys:
            entry = self.entries.get(key)
            if entry:
                self.size_tracker -= entry.size_bytes
                del self.entries[key]
                self.access_order.pop(key, None)
                self.access_counts.pop(key, None)
                self.stats.evictions += 1
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for analytics."""
        with self._lock:
            self.access_order.move_to_end(key)
            self.access_counts[key] += 1
            
            if self.enable_analytics:
                self.access_patterns[key].append(time.time())
                self.hit_patterns[key] += 1
                
                # Keep only recent access patterns
                cutoff_time = time.time() - 3600  # Last hour
                self.access_patterns[key] = [
                    t for t in self.access_patterns[key] if t > cutoff_time
                ]
    
    async def get(
        self,
        key: str,
        default: Any = None,
        update_stats: bool = True
    ) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self.entries.get(key)
            
            if entry is None:
                if update_stats:
                    self.stats.misses += 1
                return default
            
            # Check expiration
            if entry.is_expired:
                self._evict_entries([key])
                if update_stats:
                    self.stats.misses += 1
                return default
            
            # Update access tracking
            entry.last_accessed = time.time()
            self._update_access_tracking(key)
            
            if update_stats:
                self.stats.hits += 1
            
            # Deserialize value
            try:
                value = self._deserialize_value(entry.value, entry.compressed)
                return value
            except Exception:
                # Corrupted entry, remove it
                self._evict_entries([key])
                if update_stats:
                    self.stats.misses += 1
                return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        privacy_level: str = "public",
        user_id: Optional[str] = None,
        department: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with privacy awareness."""
        # Check if should cache
        if not self.privacy_manager.should_cache(privacy_level, user_id or "", department or ""):
            return False
        
        # Calculate privacy-aware TTL
        if ttl is not None:
            ttl = self.privacy_manager.calculate_ttl(
                ttl, privacy_level, department or "general"
            )
            if ttl <= 0:
                return False
        
        # Serialize and compress value
        try:
            serialized_value, compressed = self._serialize_value(value)
        except Exception:
            return False
        
        entry_size = len(serialized_value)
        
        with self._lock:
            # Check if we need to evict
            while self._should_evict() or (self.size_tracker + entry_size > self.max_size_bytes):
                if not self.entries:
                    break
                
                candidates = self._select_eviction_candidates(
                    max(1, len(self.entries) // 10)  # Evict up to 10% of entries
                )
                self._evict_entries(candidates)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                privacy_level=privacy_level,
                user_id=user_id,
                department=department,
                compressed=compressed,
                size_bytes=entry_size,
                tags=tags or []
            )
            
            # Remove old entry if exists
            old_entry = self.entries.get(key)
            if old_entry:
                self.size_tracker -= old_entry.size_bytes
            
            # Add new entry
            self.entries[key] = entry
            self.access_order[key] = None
            self.size_tracker += entry_size
            self.stats.insertions += 1
            self.stats.size_bytes = self.size_tracker
            self.stats.entry_count = len(self.entries)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self.entries:
                self._evict_entries([key])
                return True
            return False
    
    async def clear(self, privacy_level: Optional[str] = None, department: Optional[str] = None):
        """Clear cache entries based on criteria."""
        with self._lock:
            if privacy_level is None and department is None:
                # Clear all
                self.entries.clear()
                self.access_order.clear()
                self.access_counts.clear()
                self.size_tracker = 0
            else:
                # Selective clear
                keys_to_remove = []
                for key, entry in self.entries.items():
                    if (privacy_level is None or entry.privacy_level == privacy_level) and \
                       (department is None or entry.department == department):
                        keys_to_remove.append(key)
                
                self._evict_entries(keys_to_remove)
            
            self.stats.size_bytes = self.size_tracker
            self.stats.entry_count = len(self.entries)
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired
            ]
            
            self._evict_entries(expired_keys)
            self.stats.size_bytes = self.size_tracker
            self.stats.entry_count = len(self.entries)
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                insertions=self.stats.insertions,
                size_bytes=self.size_tracker,
                entry_count=len(self.entries)
            )
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self._lock:
            stats = self.get_stats()
            
            # Privacy level breakdown
            privacy_breakdown = defaultdict(int)
            department_breakdown = defaultdict(int)
            size_breakdown = defaultdict(int)
            
            for entry in self.entries.values():
                privacy_breakdown[entry.privacy_level] += 1
                if entry.department:
                    department_breakdown[entry.department] += 1
                size_breakdown[entry.privacy_level] += entry.size_bytes
            
            # Access pattern analysis
            hot_keys = []
            if self.enable_analytics:
                for key, count in self.hit_patterns.items():
                    if count > 10:  # Threshold for "hot" data
                        hot_keys.append((key, count))
                hot_keys.sort(key=lambda x: x[1], reverse=True)
                hot_keys = hot_keys[:10]  # Top 10
            
            return {
                "basic_stats": asdict(stats),
                "hit_rate": stats.hit_rate,
                "miss_rate": stats.miss_rate,
                "memory_usage_mb": self.size_tracker / (1024 * 1024),
                "memory_utilization": self.size_tracker / self.max_size_bytes,
                "average_entry_size": self.size_tracker / len(self.entries) if self.entries else 0,
                "privacy_breakdown": dict(privacy_breakdown),
                "department_breakdown": dict(department_breakdown),
                "size_by_privacy": {k: v / (1024 * 1024) for k, v in size_breakdown.items()},
                "hot_keys": hot_keys,
                "eviction_policy": self.eviction_policy.value,
                "compression_enabled": self.enable_compression
            }
    
    async def warm_cache(self, warming_function: Callable, keys: List[str]):
        """Warm cache with pre-computed values."""
        if not self.warm_cache_enabled:
            return
        
        async def warm_key(key):
            try:
                # Check if already cached
                if await self.get(key, update_stats=False) is not None:
                    return
                
                # Compute value
                value = await warming_function(key)
                if value is not None:
                    await self.set(key, value, ttl=3600)  # 1 hour default TTL
            
            except Exception:
                # Ignore warming errors
                pass
        
        # Warm keys in parallel
        warming_tasks = [warm_key(key) for key in keys]
        await asyncio.gather(*warming_tasks, return_exceptions=True)
    
    def optimize_performance(self):
        """Optimize cache performance based on usage patterns."""
        if not self.enable_analytics:
            return
        
        with self._lock:
            # Analyze access patterns
            total_accesses = sum(self.hit_patterns.values())
            if total_accesses < 100:  # Not enough data
                return
            
            # Calculate hot vs cold data ratio
            hot_threshold = total_accesses * 0.8 / len(self.hit_patterns)  # 80/20 rule
            hot_keys = [key for key, count in self.hit_patterns.items() if count > hot_threshold]
            
            # Adjust eviction policy if needed
            hot_ratio = len(hot_keys) / len(self.entries) if self.entries else 0
            
            if hot_ratio > 0.3:  # Lots of hot data
                if self.eviction_policy != EvictionPolicy.LFU:
                    self.eviction_policy = EvictionPolicy.LFU
            elif hot_ratio < 0.1:  # Mostly cold data
                if self.eviction_policy != EvictionPolicy.LRU:
                    self.eviction_policy = EvictionPolicy.LRU
            else:
                if self.eviction_policy != EvictionPolicy.ADAPTIVE:
                    self.eviction_policy = EvictionPolicy.ADAPTIVE


class CacheManager:
    """Manages multiple cache instances and provides unified interface."""
    
    def __init__(self):
        # Different cache instances for different use cases
        self.caches = {
            "inference": IntelligentCache(
                max_size_mb=256,
                eviction_policy=EvictionPolicy.ADAPTIVE
            ),
            "model": IntelligentCache(
                max_size_mb=1024,
                eviction_policy=EvictionPolicy.LFU
            ),
            "privacy": IntelligentCache(
                max_size_mb=64,
                eviction_policy=EvictionPolicy.PRIVACY_TTL
            ),
            "session": IntelligentCache(
                max_size_mb=32,
                eviction_policy=EvictionPolicy.TTL
            )
        }
        
        # Start background maintenance
        self._maintenance_task = None
        self._start_maintenance()
    
    def _start_maintenance(self):
        """Start background maintenance tasks."""
        async def maintenance_loop():
            while True:
                try:
                    # Clean up expired entries
                    for cache in self.caches.values():
                        await cache.cleanup_expired()
                    
                    # Optimize performance
                    for cache in self.caches.values():
                        cache.optimize_performance()
                    
                    await asyncio.sleep(300)  # Every 5 minutes
                
                except Exception:
                    await asyncio.sleep(60)  # Retry in 1 minute on error
        
        self._maintenance_task = asyncio.create_task(maintenance_loop())
    
    async def get(self, cache_type: str, key: str, default: Any = None) -> Any:
        """Get value from specific cache."""
        cache = self.caches.get(cache_type)
        if cache:
            return await cache.get(key, default)
        return default
    
    async def set(
        self,
        cache_type: str,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Set value in specific cache."""
        cache = self.caches.get(cache_type)
        if cache:
            return await cache.set(key, value, ttl=ttl, **kwargs)
        return False
    
    async def delete(self, cache_type: str, key: str) -> bool:
        """Delete from specific cache."""
        cache = self.caches.get(cache_type)
        if cache:
            return await cache.delete(key)
        return False
    
    async def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            await cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            name: cache.get_detailed_stats()
            for name, cache in self.caches.items()
        }
    
    async def shutdown(self):
        """Shutdown cache manager."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass


# Decorator for caching function results
def cached(
    cache_type: str = "inference",
    ttl: Optional[float] = 3600,
    privacy_level: str = "public",
    key_generator: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_manager = _get_global_cache_manager()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(
                cache_type=cache_type,
                key=cache_key,
                value=result,
                ttl=ttl,
                privacy_level=privacy_level
            )
            
            return result
        
        return wrapper
    return decorator


# Global cache manager
_global_cache_manager: Optional[CacheManager] = None


def _get_global_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    return _get_global_cache_manager()