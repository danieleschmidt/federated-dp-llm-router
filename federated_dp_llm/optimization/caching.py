"""
Advanced Caching System

Implements intelligent caching with privacy-aware cache policies, adaptive
TTL, and multi-tier caching for federated LLM systems.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from abc import ABC, abstractmethod
import logging

# Conditional redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


class CacheLevel(Enum):
    """Cache levels in multi-tier architecture."""
    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"    # Redis cache
    L3_DISK = "l3_disk"      # Disk-based cache


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    PRIVACY_AWARE = "privacy_aware"  # Custom privacy-based eviction


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    privacy_level: str = "public"  # public, sensitive, protected
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all values from cache."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in cache."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.data: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        async with self._lock:
            entry = self.data.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                await self._remove_key(key)
                return None
            
            # Update access tracking
            entry.update_access()
            self._update_access_order(key)
            
            return entry
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            # Evict if at capacity
            if len(self.data) >= self.max_size and key not in self.data:
                await self._evict_lru()
            
            # Calculate size
            entry.size_bytes = len(pickle.dumps(entry.value))
            
            self.data[key] = entry
            self._update_access_order(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            return await self._remove_key(key)
    
    async def clear(self) -> bool:
        """Clear all values from memory cache."""
        async with self._lock:
            self.data.clear()
            self.access_order.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        async with self._lock:
            if pattern == "*":
                return list(self.data.keys())
            return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def size(self) -> int:
        """Get number of entries in cache."""
        return len(self.data)
    
    def _update_access_order(self, key: str):
        """Update LRU access order."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    async def _remove_key(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self.data:
            del self.data[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    async def _evict_lru(self):
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order[0]
            await self._remove_key(lru_key)


class RedisCache(CacheBackend):
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 prefix: str = "federated_dp:", default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available. Install redis-py to use Redis caching.")
        
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.redis_client: Optional['redis.Redis'] = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_client(self) -> 'redis.Redis':
        """Get Redis client, creating if necessary."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # We handle encoding ourselves
            )
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Create prefixed Redis key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from Redis cache."""
        try:
            client = await self._get_client()
            redis_key = self._make_key(key)
            
            data = await client.get(redis_key)
            if data is None:
                return None
            
            # Deserialize cache entry
            entry_dict = pickle.loads(data)
            entry = CacheEntry(**entry_dict)
            
            if entry.is_expired():
                await self.delete(key)
                return None
            
            # Update access tracking
            entry.update_access()
            
            # Update in Redis (async to avoid blocking)
            asyncio.create_task(self._update_entry(key, entry))
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Redis cache get error: {e}")
            return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in Redis cache."""
        try:
            client = await self._get_client()
            redis_key = self._make_key(key)
            
            # Calculate size
            entry.size_bytes = len(pickle.dumps(entry.value))
            
            # Serialize cache entry
            entry_data = pickle.dumps(asdict(entry))
            
            # Set with TTL
            ttl = entry.ttl or self.default_ttl
            await client.setex(redis_key, int(ttl), entry_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            client = await self._get_client()
            redis_key = self._make_key(key)
            
            result = await client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all values from Redis cache."""
        try:
            client = await self._get_client()
            
            # Get all keys with prefix
            keys = await client.keys(f"{self.prefix}*")
            if keys:
                await client.delete(*keys)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis cache clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        try:
            client = await self._get_client()
            redis_pattern = f"{self.prefix}{pattern}"
            
            keys = await client.keys(redis_pattern)
            # Remove prefix from keys
            return [key.decode() if isinstance(key, bytes) else key 
                   for key in keys]
            
        except Exception as e:
            self.logger.error(f"Redis cache keys error: {e}")
            return []
    
    async def size(self) -> int:
        """Get number of entries in cache."""
        try:
            keys = await self.keys()
            return len(keys)
        except Exception as e:
            self.logger.error(f"Redis cache size error: {e}")
            return 0
    
    async def _update_entry(self, key: str, entry: CacheEntry):
        """Update cache entry in Redis."""
        await self.set(key, entry)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class PrivacyAwareCachePolicy:
    """Privacy-aware cache eviction policy."""
    
    def __init__(self, max_sensitive_items: int = 100, 
                 sensitive_ttl: float = 1800):  # 30 minutes
        self.max_sensitive_items = max_sensitive_items
        self.sensitive_ttl = sensitive_ttl
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Determine if entry should be evicted based on privacy policy."""
        current_time = time.time()
        
        # Immediate eviction for expired sensitive data
        if entry.privacy_level in ["sensitive", "protected"]:
            if entry.is_expired():
                return True
            
            # Shorter TTL for sensitive data
            if current_time > (entry.created_at + self.sensitive_ttl):
                return True
        
        return False
    
    def calculate_priority(self, entry: CacheEntry) -> float:
        """Calculate eviction priority (higher = more likely to evict)."""
        current_time = time.time()
        priority = 0.0
        
        # Privacy level factor
        privacy_factors = {
            "public": 1.0,
            "sensitive": 3.0,
            "protected": 5.0
        }
        priority += privacy_factors.get(entry.privacy_level, 1.0)
        
        # Age factor
        age = current_time - entry.created_at
        priority += age / 3600  # Hours since creation
        
        # Access frequency factor (inverse)
        time_since_access = current_time - entry.last_accessed
        priority += time_since_access / 1800  # 30 minutes
        
        # Inverse of access count
        priority += 10.0 / max(entry.access_count, 1)
        
        return priority


class CacheManager:
    """Multi-tier cache manager with privacy awareness."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backends: Dict[CacheLevel, CacheBackend] = {}
        self.policy = PrivacyAwareCachePolicy()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0
        }
        self.logger = logging.getLogger(__name__)
        
        # Initialize backends
        self._setup_backends()
    
    def _setup_backends(self):
        """Setup cache backends based on configuration."""
        # L1 Memory cache
        if self.config.get("enable_memory_cache", True):
            memory_config = self.config.get("memory_cache", {})
            self.backends[CacheLevel.L1_MEMORY] = MemoryCache(
                max_size=memory_config.get("max_size", 10000),
                default_ttl=memory_config.get("default_ttl", 3600)
            )
        
        # L2 Redis cache
        if self.config.get("enable_redis_cache", True) and REDIS_AVAILABLE:
            redis_config = self.config.get("redis_cache", {})
            self.backends[CacheLevel.L2_REDIS] = RedisCache(
                redis_url=redis_config.get("url", "redis://localhost:6379"),
                prefix=redis_config.get("prefix", "federated_dp:"),
                default_ttl=redis_config.get("default_ttl", 3600)
            )
        elif self.config.get("enable_redis_cache", True) and not REDIS_AVAILABLE:
            self.logger.warning("Redis cache requested but redis-py not available. Falling back to memory cache only.")
    
    def _make_cache_key(self, key: str, namespace: str = "default") -> str:
        """Create cache key with namespace."""
        return f"{namespace}:{key}"
    
    def _hash_complex_key(self, key_data: Dict[str, Any]) -> str:
        """Create hash for complex key data."""
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache, checking all tiers."""
        cache_key = self._make_cache_key(key, namespace)
        
        # Check each cache level in order
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]:
            backend = self.backends.get(level)
            if backend is None:
                continue
            
            try:
                entry = await backend.get(cache_key)
                if entry is not None:
                    self.stats["hits"] += 1
                    
                    # Promote to higher cache levels
                    await self._promote_entry(cache_key, entry, level)
                    
                    return entry.value
                    
            except Exception as e:
                self.logger.error(f"Cache get error in {level}: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                  privacy_level: str = "public", tags: List[str] = None,
                  namespace: str = "default") -> bool:
        """Set value in all applicable cache tiers."""
        cache_key = self._make_cache_key(key, namespace)
        
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl=ttl,
            privacy_level=privacy_level,
            tags=tags or []
        )
        
        success = True
        
        # Set in all available backends
        for level, backend in self.backends.items():
            try:
                result = await backend.set(cache_key, entry)
                if not result:
                    success = False
                    
            except Exception as e:
                self.logger.error(f"Cache set error in {level}: {e}")
                success = False
        
        if success:
            self.stats["sets"] += 1
        
        return success
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from all cache tiers."""
        cache_key = self._make_cache_key(key, namespace)
        
        success = True
        
        for level, backend in self.backends.items():
            try:
                result = await backend.delete(cache_key)
                if not result:
                    success = False
                    
            except Exception as e:
                self.logger.error(f"Cache delete error in {level}: {e}")
                success = False
        
        if success:
            self.stats["deletes"] += 1
        
        return success
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all cache entries with specified tags."""
        invalidated = 0
        
        for level, backend in self.backends.items():
            try:
                keys = await backend.keys()
                
                for key in keys:
                    entry = await backend.get(key)
                    if entry and any(tag in entry.tags for tag in tags):
                        await backend.delete(key)
                        invalidated += 1
                        
            except Exception as e:
                self.logger.error(f"Cache invalidation error in {level}: {e}")
        
        return invalidated
    
    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all entries in a namespace."""
        pattern = f"{namespace}:*"
        
        success = True
        
        for level, backend in self.backends.items():
            try:
                keys = await backend.keys(pattern)
                for key in keys:
                    result = await backend.delete(key)
                    if not result:
                        success = False
                        
            except Exception as e:
                self.logger.error(f"Cache clear namespace error in {level}: {e}")
                success = False
        
        return success
    
    async def _promote_entry(self, key: str, entry: CacheEntry, from_level: CacheLevel):
        """Promote cache entry to higher tiers."""
        levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        from_index = levels.index(from_level)
        
        # Promote to higher levels
        for i in range(from_index):
            higher_level = levels[i]
            backend = self.backends.get(higher_level)
            
            if backend:
                try:
                    await backend.set(key, entry)
                except Exception as e:
                    self.logger.error(f"Cache promotion error to {higher_level}: {e}")
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from all cache tiers."""
        cleaned = 0
        
        for level, backend in self.backends.items():
            try:
                keys = await backend.keys()
                
                for key in keys:
                    entry = await backend.get(key)
                    if entry and entry.is_expired():
                        await backend.delete(key)
                        cleaned += 1
                        
            except Exception as e:
                self.logger.error(f"Cache cleanup error in {level}: {e}")
        
        self.stats["evictions"] += cleaned
        return cleaned
    
    async def privacy_cleanup(self) -> int:
        """Clean up sensitive data based on privacy policy."""
        cleaned = 0
        
        for level, backend in self.backends.items():
            try:
                keys = await backend.keys()
                
                for key in keys:
                    entry = await backend.get(key)
                    if entry and self.policy.should_evict(entry):
                        await backend.delete(key)
                        cleaned += 1
                        
            except Exception as e:
                self.logger.error(f"Privacy cleanup error in {level}: {e}")
        
        self.stats["evictions"] += cleaned
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total_requests, 1)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "enabled_backends": list(self.backends.keys())
        }
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        info = {
            "stats": self.get_stats(),
            "backends": {}
        }
        
        for level, backend in self.backends.items():
            try:
                backend_info = {
                    "size": await backend.size(),
                    "type": type(backend).__name__
                }
                
                # Additional info for specific backends
                if isinstance(backend, MemoryCache):
                    backend_info.update({
                        "max_size": backend.max_size,
                        "access_order_length": len(backend.access_order)
                    })
                elif isinstance(backend, RedisCache):
                    backend_info.update({
                        "redis_url": backend.redis_url,
                        "prefix": backend.prefix
                    })
                
                info["backends"][level.value] = backend_info
                
            except Exception as e:
                self.logger.error(f"Error getting cache info for {level}: {e}")
                info["backends"][level.value] = {"error": str(e)}
        
        return info
    
    async def close(self):
        """Close all cache backends."""
        for backend in self.backends.values():
            if hasattr(backend, 'close'):
                await backend.close()