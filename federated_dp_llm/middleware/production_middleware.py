"""
Production-Ready Middleware Suite for Federated DP-LLM Router

Implements comprehensive middleware stack including request/response handling,
caching, compression, rate limiting, and performance optimization for healthcare
federated learning environments.
"""

import asyncio
import time
import gzip
import json
import hashlib
import logging
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
from functools import wraps


logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Response compression levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


class CacheStrategy(Enum):
    """Caching strategies."""
    NO_CACHE = "no-cache"
    PRIVATE = "private"
    PUBLIC = "public"
    PRIVACY_AWARE = "privacy-aware"  # Special healthcare-aware caching


@dataclass
class MiddlewareConfig:
    """Configuration for production middleware."""
    # Compression settings
    enable_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    min_compression_size: int = 1024  # bytes
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes default
    max_cache_size: int = 1000  # number of entries
    cache_strategy: CacheStrategy = CacheStrategy.PRIVACY_AWARE
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    burst_limit: int = 10
    
    # Request/Response settings
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    request_timeout: int = 30  # seconds
    
    # Performance settings
    enable_request_batching: bool = True
    batch_size: int = 10
    batch_timeout: float = 0.1  # 100ms
    
    # Health and monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_sensitive_data: bool = False  # HIPAA compliance


@dataclass
class RequestMetrics:
    """Request metrics for monitoring."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    method: str = ""
    path: str = ""
    status_code: int = 200
    request_size: int = 0
    response_size: int = 0
    user_id: Optional[str] = None
    cache_hit: bool = False
    compressed: bool = False
    processing_time: float = 0.0
    database_time: float = 0.0
    external_api_time: float = 0.0


class RateLimiter:
    """Advanced rate limiter with privacy-aware features."""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.client_requests = defaultdict(lambda: deque(maxlen=burst_limit * 2))
        self.user_requests = defaultdict(lambda: deque(maxlen=requests_per_minute * 2))
        
    async def is_allowed(self, client_ip: str, user_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        client_requests = self.client_requests[client_ip]
        while client_requests and client_requests[0] < minute_ago:
            client_requests.popleft()
        
        # Check per-IP rate limit
        if len(client_requests) >= self.requests_per_minute:
            return False, {"reason": "IP rate limit exceeded", "retry_after": 60}
        
        # Check burst limit
        recent_requests = [req for req in client_requests if current_time - req < 10]  # Last 10 seconds
        if len(recent_requests) >= self.burst_limit:
            return False, {"reason": "Burst limit exceeded", "retry_after": 10}
        
        # Check user-specific limits (if user authenticated)
        if user_id:
            user_requests = self.user_requests[user_id]
            while user_requests and user_requests[0] < minute_ago:
                user_requests.popleft()
            
            # Healthcare users get higher limits
            user_limit = self.requests_per_minute * 2 if self._is_healthcare_user(user_id) else self.requests_per_minute
            
            if len(user_requests) >= user_limit:
                return False, {"reason": "User rate limit exceeded", "retry_after": 60}
            
            user_requests.append(current_time)
        
        # Record request
        client_requests.append(current_time)
        
        return True, {
            "remaining_requests": self.requests_per_minute - len(client_requests),
            "reset_time": minute_ago + 60
        }
    
    def _is_healthcare_user(self, user_id: str) -> bool:
        """Check if user is a healthcare professional (higher rate limits)."""
        # In production, this would check user roles/permissions
        return user_id.startswith(('doctor_', 'nurse_', 'clinician_'))


class PrivacyAwareCache:
    """Privacy-aware caching system for healthcare data."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.privacy_levels = {}
        
    def _generate_cache_key(self, request_data: Dict, user_context: Dict) -> str:
        """Generate privacy-aware cache key."""
        # Remove sensitive data before hashing
        sanitized_data = self._sanitize_for_cache(request_data)
        
        # Include user context but not sensitive identifiers
        context_data = {
            "role": user_context.get("role", ""),
            "department": user_context.get("department", ""),
            "privacy_level": user_context.get("privacy_level", "standard")
        }
        
        cache_data = {"request": sanitized_data, "context": context_data}
        return hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _sanitize_for_cache(self, data: Dict) -> Dict:
        """Remove sensitive information before caching."""
        sanitized = {}
        sensitive_keys = {
            'ssn', 'social_security_number', 'patient_id', 'mrn', 'medical_record_number',
            'dob', 'date_of_birth', 'phone', 'email', 'address', 'insurance_id'
        }
        
        for key, value in data.items():
            if key.lower() in sensitive_keys:
                continue  # Skip sensitive data
            
            if isinstance(value, str) and len(value) > 100:
                # Hash long text fields that might contain PHI
                sanitized[key] = hashlib.sha256(value.encode()).hexdigest()[:16]
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def get(self, request_data: Dict, user_context: Dict) -> Optional[Any]:
        """Get cached response if available and privacy-compliant."""
        cache_key = self._generate_cache_key(request_data, user_context)
        
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        current_time = time.time()
        
        # Check TTL
        if current_time - cached_item['timestamp'] > cached_item['ttl']:
            del self.cache[cache_key]
            return None
        
        # Check privacy compliance
        if not self._is_privacy_compliant(cached_item, user_context):
            return None
        
        # Update access time
        self.access_times[cache_key] = current_time
        
        logger.debug(f"Cache hit for key: {cache_key[:8]}...")
        return cached_item['data']
    
    async def set(self, request_data: Dict, user_context: Dict, response_data: Any, ttl: Optional[int] = None):
        """Cache response data with privacy considerations."""
        # Don't cache responses containing sensitive data
        if self._contains_sensitive_data(response_data):
            logger.debug("Skipping cache due to sensitive data")
            return
        
        cache_key = self._generate_cache_key(request_data, user_context)
        current_time = time.time()
        
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size:
            await self._evict_oldest()
        
        # Determine TTL based on data sensitivity
        cache_ttl = ttl or self._calculate_privacy_ttl(response_data, user_context)
        
        self.cache[cache_key] = {
            'data': response_data,
            'timestamp': current_time,
            'ttl': cache_ttl,
            'privacy_level': user_context.get('privacy_level', 'standard'),
            'user_role': user_context.get('role', 'unknown')
        }
        
        self.access_times[cache_key] = current_time
        logger.debug(f"Cached response for key: {cache_key[:8]}... (TTL: {cache_ttl}s)")
    
    def _contains_sensitive_data(self, data: Any) -> bool:
        """Check if response contains sensitive healthcare data."""
        if isinstance(data, dict):
            sensitive_keys = {'patient_data', 'phi', 'personal_health_information', 'diagnosis'}
            return any(key in str(data).lower() for key in sensitive_keys)
        elif isinstance(data, str):
            sensitive_patterns = ['patient:', 'diagnosis:', 'medication:', 'treatment:']
            return any(pattern in data.lower() for pattern in sensitive_patterns)
        return False
    
    def _is_privacy_compliant(self, cached_item: Dict, user_context: Dict) -> bool:
        """Check if cached item can be served to current user context."""
        cached_privacy_level = cached_item.get('privacy_level', 'standard')
        current_privacy_level = user_context.get('privacy_level', 'standard')
        
        # Privacy levels: public < standard < restricted < confidential
        privacy_hierarchy = {'public': 0, 'standard': 1, 'restricted': 2, 'confidential': 3}
        
        cached_level = privacy_hierarchy.get(cached_privacy_level, 1)
        current_level = privacy_hierarchy.get(current_privacy_level, 1)
        
        return current_level >= cached_level
    
    def _calculate_privacy_ttl(self, data: Any, user_context: Dict) -> int:
        """Calculate TTL based on data sensitivity and user context."""
        base_ttl = self.default_ttl
        
        # Reduce TTL for sensitive data
        if self._contains_sensitive_data(data):
            base_ttl = min(base_ttl, 60)  # Max 1 minute for sensitive data
        
        # Adjust based on user privacy level
        privacy_level = user_context.get('privacy_level', 'standard')
        if privacy_level == 'confidential':
            base_ttl = min(base_ttl, 30)  # 30 seconds for confidential users
        elif privacy_level == 'restricted':
            base_ttl = min(base_ttl, 120)  # 2 minutes for restricted users
        
        return base_ttl
    
    async def _evict_oldest(self):
        """Evict least recently used cache entries."""
        if not self.access_times:
            return
        
        # Find oldest accessed item
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        del self.access_times[oldest_key]
        
        logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")


class ResponseCompressor:
    """Response compression with healthcare-aware settings."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        
    async def compress_response(self, data: bytes, content_type: str = "") -> Tuple[bytes, Dict[str, str]]:
        """Compress response data if beneficial."""
        if not self.config.enable_compression:
            return data, {}
        
        # Skip compression for small responses
        if len(data) < self.config.min_compression_size:
            return data, {}
        
        # Skip compression for already compressed content
        if content_type.startswith(('image/', 'video/', 'audio/')):
            return data, {}
        
        try:
            compressed_data = gzip.compress(data, compresslevel=self.config.compression_level.value)
            
            # Only use compressed version if it's significantly smaller
            if len(compressed_data) < len(data) * 0.9:
                headers = {
                    'Content-Encoding': 'gzip',
                    'Content-Length': str(len(compressed_data))
                }
                logger.debug(f"Compressed response: {len(data)} -> {len(compressed_data)} bytes")
                return compressed_data, headers
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
        
        return data, {}


class RequestBatcher:
    """Intelligent request batching for improved efficiency."""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self.pending_requests = {}
        self.batch_timers = {}
        
    async def add_request(self, request_id: str, request_data: Dict, callback: Callable) -> bool:
        """Add request to batch if possible."""
        if not self.config.enable_request_batching:
            return False
        
        # Check if request can be batched
        batch_key = self._get_batch_key(request_data)
        if not batch_key:
            return False
        
        # Initialize batch if needed
        if batch_key not in self.pending_requests:
            self.pending_requests[batch_key] = []
            
            # Set batch timer
            self.batch_timers[batch_key] = asyncio.create_task(
                self._process_batch_after_timeout(batch_key)
            )
        
        # Add request to batch
        self.pending_requests[batch_key].append({
            'request_id': request_id,
            'data': request_data,
            'callback': callback,
            'timestamp': time.time()
        })
        
        # Process batch if full
        if len(self.pending_requests[batch_key]) >= self.config.batch_size:
            await self._process_batch(batch_key)
        
        return True
    
    def _get_batch_key(self, request_data: Dict) -> Optional[str]:
        """Determine if request can be batched and generate batch key."""
        # Only batch similar model inference requests
        if request_data.get('type') != 'inference':
            return None
        
        # Create batch key based on model and similar parameters
        model_name = request_data.get('model', 'default')
        privacy_params = request_data.get('privacy_params', {})
        epsilon = privacy_params.get('epsilon', 1.0)
        
        # Round epsilon to create batch groups
        epsilon_group = round(epsilon, 1)
        
        return f"inference_{model_name}_{epsilon_group}"
    
    async def _process_batch_after_timeout(self, batch_key: str):
        """Process batch after timeout expires."""
        await asyncio.sleep(self.config.batch_timeout)
        await self._process_batch(batch_key)
    
    async def _process_batch(self, batch_key: str):
        """Process accumulated batch requests."""
        if batch_key not in self.pending_requests:
            return
        
        requests = self.pending_requests[batch_key]
        if not requests:
            return
        
        # Cancel timer
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()
            del self.batch_timers[batch_key]
        
        logger.debug(f"Processing batch {batch_key} with {len(requests)} requests")
        
        # Process batch (mock implementation)
        try:
            # In production, this would call the actual batch processing logic
            results = await self._process_batch_requests(requests)
            
            # Return results to individual callbacks
            for request, result in zip(requests, results):
                await request['callback'](result)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            # Return error to all callbacks
            for request in requests:
                await request['callback']({"error": "Batch processing failed"})
        
        # Clean up
        del self.pending_requests[batch_key]
    
    async def _process_batch_requests(self, requests: List[Dict]) -> List[Dict]:
        """Process batch of similar requests efficiently."""
        # Mock batch processing - in production this would optimize inference
        results = []
        for request in requests:
            # Simulate processing
            await asyncio.sleep(0.01)  # Simulate work
            results.append({
                'request_id': request['request_id'],
                'result': f"Processed: {request['data'].get('prompt', 'Unknown')[:50]}...",
                'batched': True
            })
        
        return results


class ProductionMiddlewareStack:
    """Complete middleware stack for production deployment."""
    
    def __init__(self, config: MiddlewareConfig = None):
        self.config = config or MiddlewareConfig()
        self.rate_limiter = RateLimiter(
            self.config.requests_per_minute,
            self.config.burst_limit
        )
        self.cache = PrivacyAwareCache(
            self.config.max_cache_size,
            self.config.cache_ttl
        )
        self.compressor = ResponseCompressor(self.config)
        self.batcher = RequestBatcher(self.config)
        self.request_metrics = {}
        
    async def process_request(self, request_data: Dict, user_context: Dict) -> Dict[str, Any]:
        """Process incoming request through middleware stack."""
        request_id = request_data.get('request_id', f"req_{int(time.time())}")
        start_time = time.time()
        
        # Initialize request metrics
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=start_time,
            method=request_data.get('method', 'POST'),
            path=request_data.get('path', '/'),
            user_id=user_context.get('user_id'),
            request_size=len(json.dumps(request_data, default=str))
        )
        self.request_metrics[request_id] = metrics
        
        try:
            # 1. Rate limiting
            client_ip = request_data.get('client_ip', 'unknown')
            allowed, rate_info = await self.rate_limiter.is_allowed(
                client_ip, user_context.get('user_id')
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_ip}: {rate_info['reason']}")
                return {
                    'error': 'Rate limit exceeded',
                    'retry_after': rate_info.get('retry_after', 60),
                    'status_code': 429
                }
            
            # 2. Request validation
            if metrics.request_size > self.config.max_request_size:
                logger.warning(f"Request too large: {metrics.request_size} bytes")
                return {
                    'error': 'Request too large',
                    'max_size': self.config.max_request_size,
                    'status_code': 413
                }
            
            # 3. Cache check
            cached_response = await self.cache.get(request_data, user_context)
            if cached_response:
                metrics.cache_hit = True
                metrics.end_time = time.time()
                metrics.processing_time = metrics.end_time - metrics.start_time
                
                logger.debug(f"Cache hit for request {request_id}")
                return {
                    'data': cached_response,
                    'cached': True,
                    'status_code': 200
                }
            
            # 4. Request processing (placeholder)
            response_data = await self._process_core_request(request_data, user_context)
            
            # 5. Cache response
            if response_data.get('status_code', 200) == 200:
                await self.cache.set(request_data, user_context, response_data.get('data'))
            
            # 6. Update metrics
            metrics.end_time = time.time()
            metrics.processing_time = metrics.end_time - metrics.start_time
            metrics.status_code = response_data.get('status_code', 200)
            metrics.response_size = len(json.dumps(response_data, default=str))
            
            return response_data
            
        except Exception as e:
            logger.error(f"Middleware processing failed for {request_id}: {str(e)}")
            metrics.end_time = time.time()
            metrics.status_code = 500
            
            return {
                'error': 'Internal server error',
                'status_code': 500
            }
    
    async def _process_core_request(self, request_data: Dict, user_context: Dict) -> Dict[str, Any]:
        """Core request processing (to be implemented by actual service)."""
        # This is a placeholder - in production this would call the actual
        # federated DP-LLM processing logic
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'data': {
                'result': f"Processed request: {request_data.get('prompt', 'Unknown')[:50]}...",
                'model_used': request_data.get('model', 'default'),
                'privacy_budget_used': request_data.get('privacy_params', {}).get('epsilon', 1.0)
            },
            'status_code': 200,
            'processing_time': 0.1
        }
    
    async def compress_response(self, response_data: bytes, content_type: str = "application/json") -> Tuple[bytes, Dict[str, str]]:
        """Compress response if beneficial."""
        return await self.compressor.compress_response(response_data, content_type)
    
    def get_metrics_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance metrics summary."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_metrics = [
            metrics for metrics in self.request_metrics.values()
            if metrics.start_time >= cutoff_time and metrics.end_time
        ]
        
        if not recent_metrics:
            return {"message": "No recent requests"}
        
        # Calculate statistics
        processing_times = [m.processing_time for m in recent_metrics]
        response_sizes = [m.response_size for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        compressed_responses = sum(1 for m in recent_metrics if m.compressed)
        
        status_codes = {}
        for metrics in recent_metrics:
            status_codes[metrics.status_code] = status_codes.get(metrics.status_code, 0) + 1
        
        return {
            "time_window_hours": time_window / 3600,
            "total_requests": len(recent_metrics),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "median_processing_time": sorted(processing_times)[len(processing_times) // 2],
            "cache_hit_rate": cache_hits / len(recent_metrics),
            "compression_rate": compressed_responses / len(recent_metrics),
            "average_response_size": sum(response_sizes) / len(response_sizes),
            "status_code_distribution": status_codes,
            "error_rate": sum(1 for code in status_codes.keys() if code >= 400) / len(recent_metrics)
        }


# Global middleware instance
production_middleware = ProductionMiddlewareStack()


# FastAPI middleware integration
async def production_middleware_handler(request, call_next):
    """FastAPI middleware handler."""
    # Extract request data
    request_data = {
        'request_id': f"req_{int(time.time() * 1000000)}",
        'method': request.method,
        'path': str(request.url.path),
        'client_ip': request.client.host if request.client else 'unknown',
        'headers': dict(request.headers),
        'query_params': dict(request.query_params)
    }
    
    # Extract user context
    user_context = {
        'user_id': request.headers.get('user-id'),
        'role': request.headers.get('user-role', 'standard'),
        'department': request.headers.get('user-department'),
        'privacy_level': request.headers.get('privacy-level', 'standard')
    }
    
    # Process through middleware
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    # Add performance headers
    response.headers['X-Processing-Time'] = str(processing_time)
    response.headers['X-Request-ID'] = request_data['request_id']
    
    return response